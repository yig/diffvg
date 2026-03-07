import torch
import pydiffvg
import time
from enum import IntEnum
import warnings
from . import torch_render

print_timing = False

def set_print_timing(val):
    global print_timing
    print_timing = val

class OutputType(IntEnum):
    color = 1
    sdf = 2

class RenderFunction(torch.autograd.Function):
    """
    Pure PyTorch differentiable vector graphics renderer.
    API-compatible replacement for the C++/CUDA backend.
    Works on any PyTorch device (CPU, CUDA, MPS/Metal, ROCm).
    """

    @staticmethod
    def serialize_scene(canvas_width,
                        canvas_height,
                        shapes,
                        shape_groups,
                        filter=None,
                        output_type=OutputType.color,
                        use_prefiltering=False,
                        eval_positions=torch.tensor([])):
        """Convert shapes/groups to a flat argument list for PyTorch autograd."""
        if filter is None:
            filter = pydiffvg.PixelFilter(type=torch_render.FilterType.box,
                                          radius=torch.tensor(0.5))

        num_shapes = len(shapes)
        num_shape_groups = len(shape_groups)
        args = []
        args.append(canvas_width)
        args.append(canvas_height)
        args.append(num_shapes)
        args.append(num_shape_groups)
        args.append(output_type)
        args.append(use_prefiltering)
        args.append(eval_positions)

        for shape in shapes:
            use_thickness = False
            if isinstance(shape, pydiffvg.Circle):
                args.append(torch_render.ShapeType.circle)
                args.append(shape.radius if isinstance(shape.radius, torch.Tensor) else torch.tensor(float(shape.radius)))
                args.append(shape.center)
            elif isinstance(shape, pydiffvg.Ellipse):
                args.append(torch_render.ShapeType.ellipse)
                args.append(shape.radius)
                args.append(shape.center)
            elif isinstance(shape, pydiffvg.Path):
                args.append(torch_render.ShapeType.path)
                args.append(shape.num_control_points.to(torch.int32))
                args.append(shape.points)
                if len(shape.stroke_width.shape) > 0 and shape.stroke_width.shape[0] > 1:
                    use_thickness = True
                    args.append(shape.stroke_width)
                else:
                    args.append(None)
                args.append(shape.is_closed)
                args.append(getattr(shape, 'use_distance_approx', False))
            elif isinstance(shape, pydiffvg.Polygon):
                args.append(torch_render.ShapeType.path)
                if shape.is_closed:
                    args.append(torch.zeros(shape.points.shape[0], dtype=torch.int32))
                else:
                    args.append(torch.zeros(shape.points.shape[0] - 1, dtype=torch.int32))
                args.append(shape.points)
                args.append(None)
                args.append(shape.is_closed)
                args.append(False)
            elif isinstance(shape, pydiffvg.Rect):
                args.append(torch_render.ShapeType.rect)
                args.append(shape.p_min)
                args.append(shape.p_max)
            else:
                assert False, f"Unknown shape type: {type(shape)}"
            if use_thickness:
                args.append(torch.tensor(0.0))
            else:
                args.append(shape.stroke_width if isinstance(shape.stroke_width, torch.Tensor)
                           else torch.tensor(float(shape.stroke_width)))

        for shape_group in shape_groups:
            args.append(shape_group.shape_ids.to(torch.int32))
            # Fill color
            if shape_group.fill_color is None:
                args.append(None)
            elif isinstance(shape_group.fill_color, torch.Tensor):
                args.append(torch_render.ColorType.constant)
                args.append(shape_group.fill_color)
            elif isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                args.append(torch_render.ColorType.linear_gradient)
                args.append(shape_group.fill_color.begin)
                args.append(shape_group.fill_color.end)
                args.append(shape_group.fill_color.offsets)
                args.append(shape_group.fill_color.stop_colors)
            elif isinstance(shape_group.fill_color, pydiffvg.RadialGradient):
                args.append(torch_render.ColorType.radial_gradient)
                args.append(shape_group.fill_color.center)
                args.append(shape_group.fill_color.radius)
                args.append(shape_group.fill_color.offsets)
                args.append(shape_group.fill_color.stop_colors)

            if shape_group.fill_color is not None:
                for shape_id in shape_group.shape_ids:
                    if isinstance(shapes[shape_id], pydiffvg.Path):
                        if not shapes[shape_id].is_closed:
                            warnings.warn("Detected non-closed paths with fill color. "
                                        "This might cause unexpected results.", Warning)

            # Stroke color
            if shape_group.stroke_color is None:
                args.append(None)
            elif isinstance(shape_group.stroke_color, torch.Tensor):
                args.append(torch_render.ColorType.constant)
                args.append(shape_group.stroke_color)
            elif isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
                args.append(torch_render.ColorType.linear_gradient)
                args.append(shape_group.stroke_color.begin)
                args.append(shape_group.stroke_color.end)
                args.append(shape_group.stroke_color.offsets)
                args.append(shape_group.stroke_color.stop_colors)
            elif isinstance(shape_group.stroke_color, pydiffvg.RadialGradient):
                args.append(torch_render.ColorType.radial_gradient)
                args.append(shape_group.stroke_color.center)
                args.append(shape_group.stroke_color.radius)
                args.append(shape_group.stroke_color.offsets)
                args.append(shape_group.stroke_color.stop_colors)
            args.append(shape_group.use_even_odd_rule)
            args.append(shape_group.shape_to_canvas.contiguous())
        args.append(filter.type)
        args.append(filter.radius if isinstance(filter.radius, torch.Tensor) else torch.tensor(float(filter.radius)))
        return args

    @staticmethod
    def forward(ctx, width, height, num_samples_x, num_samples_y, seed,
                background_image, *args):
        """Forward rendering pass using pure PyTorch."""
        device = pydiffvg.get_device()

        # Unpack arguments
        current_index = 0
        canvas_width = args[current_index]; current_index += 1
        canvas_height = args[current_index]; current_index += 1
        num_shapes = args[current_index]; current_index += 1
        num_shape_groups = args[current_index]; current_index += 1
        output_type = args[current_index]; current_index += 1
        use_prefiltering = args[current_index]; current_index += 1
        eval_positions = args[current_index]; current_index += 1

        # Parse shapes
        shapes = []
        tensor_args = []  # Track tensor args for gradient

        for shape_id in range(num_shapes):
            shape_type = args[current_index]; current_index += 1

            if shape_type == torch_render.ShapeType.circle:
                radius = args[current_index]; current_index += 1
                center = args[current_index]; current_index += 1
                r = radius.to(device) if isinstance(radius, torch.Tensor) else torch.tensor(float(radius), device=device)
                c = center.to(device)
                data = {'center': c, 'radius': r}
                tensor_args.extend([r, c])
            elif shape_type == torch_render.ShapeType.ellipse:
                radius = args[current_index]; current_index += 1
                center = args[current_index]; current_index += 1
                r = radius.to(device)
                c = center.to(device)
                data = {'center': c, 'radius': r}
                tensor_args.extend([r, c])
            elif shape_type == torch_render.ShapeType.path:
                num_control_points = args[current_index]; current_index += 1
                points = args[current_index]; current_index += 1
                thickness = args[current_index]; current_index += 1
                is_closed = args[current_index]; current_index += 1
                use_distance_approx = args[current_index]; current_index += 1
                p = points.to(device)
                data = {
                    'points': p,
                    'num_control_points': num_control_points.to(device),
                    'is_closed': is_closed,
                    'use_distance_approx': use_distance_approx,
                }
                if thickness is not None:
                    data['thickness'] = thickness.to(device)
                    tensor_args.append(thickness.to(device))
                tensor_args.append(p)
            elif shape_type == torch_render.ShapeType.rect:
                p_min = args[current_index]; current_index += 1
                p_max = args[current_index]; current_index += 1
                pm = p_min.to(device)
                px = p_max.to(device)
                data = {'p_min': pm, 'p_max': px}
                tensor_args.extend([pm, px])
            else:
                assert False

            stroke_width = args[current_index]; current_index += 1
            sw = stroke_width.to(device) if isinstance(stroke_width, torch.Tensor) else torch.tensor(float(stroke_width), device=device)
            data['stroke_width'] = sw
            tensor_args.append(sw)
            shapes.append((shape_type, data))

        # Parse shape groups
        shape_groups = []
        for group_id in range(num_shape_groups):
            shape_ids = args[current_index]; current_index += 1
            group = {
                'shape_ids': shape_ids.to(device),
            }

            fill_color_type = args[current_index]; current_index += 1
            if fill_color_type == torch_render.ColorType.constant:
                color = args[current_index]; current_index += 1
                group['fill_color_type'] = torch_render.ColorType.constant
                group['fill_color_data'] = color.to(device)
                tensor_args.append(color.to(device))
            elif fill_color_type == torch_render.ColorType.linear_gradient:
                begin = args[current_index]; current_index += 1
                end = args[current_index]; current_index += 1
                offsets = args[current_index]; current_index += 1
                stop_colors = args[current_index]; current_index += 1
                group['fill_color_type'] = torch_render.ColorType.linear_gradient
                group['fill_color_data'] = {
                    'begin': begin.to(device),
                    'end': end.to(device),
                    'offsets': offsets.to(device),
                    'stop_colors': stop_colors.to(device),
                }
                tensor_args.extend([begin.to(device), end.to(device),
                                   offsets.to(device), stop_colors.to(device)])
            elif fill_color_type == torch_render.ColorType.radial_gradient:
                center = args[current_index]; current_index += 1
                radius = args[current_index]; current_index += 1
                offsets = args[current_index]; current_index += 1
                stop_colors = args[current_index]; current_index += 1
                group['fill_color_type'] = torch_render.ColorType.radial_gradient
                group['fill_color_data'] = {
                    'center': center.to(device),
                    'radius': radius.to(device),
                    'offsets': offsets.to(device),
                    'stop_colors': stop_colors.to(device),
                }
                tensor_args.extend([center.to(device), radius.to(device),
                                   offsets.to(device), stop_colors.to(device)])
            elif fill_color_type is None:
                group['fill_color_type'] = None
                group['fill_color_data'] = None
            else:
                assert False

            stroke_color_type = args[current_index]; current_index += 1
            if stroke_color_type == torch_render.ColorType.constant:
                color = args[current_index]; current_index += 1
                group['stroke_color_type'] = torch_render.ColorType.constant
                group['stroke_color_data'] = color.to(device)
                tensor_args.append(color.to(device))
            elif stroke_color_type == torch_render.ColorType.linear_gradient:
                begin = args[current_index]; current_index += 1
                end = args[current_index]; current_index += 1
                offsets = args[current_index]; current_index += 1
                stop_colors = args[current_index]; current_index += 1
                group['stroke_color_type'] = torch_render.ColorType.linear_gradient
                group['stroke_color_data'] = {
                    'begin': begin.to(device),
                    'end': end.to(device),
                    'offsets': offsets.to(device),
                    'stop_colors': stop_colors.to(device),
                }
                tensor_args.extend([begin.to(device), end.to(device),
                                   offsets.to(device), stop_colors.to(device)])
            elif stroke_color_type == torch_render.ColorType.radial_gradient:
                center = args[current_index]; current_index += 1
                radius = args[current_index]; current_index += 1
                offsets = args[current_index]; current_index += 1
                stop_colors = args[current_index]; current_index += 1
                group['stroke_color_type'] = torch_render.ColorType.radial_gradient
                group['stroke_color_data'] = {
                    'center': center.to(device),
                    'radius': radius.to(device),
                    'offsets': offsets.to(device),
                    'stop_colors': stop_colors.to(device),
                }
                tensor_args.extend([center.to(device), radius.to(device),
                                   offsets.to(device), stop_colors.to(device)])
            elif stroke_color_type is None:
                group['stroke_color_type'] = None
                group['stroke_color_data'] = None
            else:
                assert False

            use_even_odd_rule = args[current_index]; current_index += 1
            shape_to_canvas = args[current_index]; current_index += 1
            group['use_even_odd_rule'] = use_even_odd_rule
            group['shape_to_canvas'] = shape_to_canvas.to(device)
            tensor_args.append(shape_to_canvas.to(device))
            shape_groups.append(group)

        filter_type = args[current_index]; current_index += 1
        filter_radius = args[current_index]; current_index += 1
        fr = filter_radius.item() if isinstance(filter_radius, torch.Tensor) else float(filter_radius)

        # Map FilterType from C++ enum to our enum
        ft = _convert_filter_type(filter_type)

        start = time.time()

        if output_type == OutputType.color:
            rendered_image = torch_render.render_scene(
                width, height, num_samples_x, num_samples_y, seed,
                canvas_width, canvas_height,
                shapes, shape_groups,
                ft, fr,
                background_image=background_image,
                use_prefiltering=use_prefiltering,
                device=device)
        else:
            # SDF mode
            rendered_image = _render_sdf(
                width, height, num_samples_x, num_samples_y, seed,
                canvas_width, canvas_height,
                shapes, shape_groups, device, eval_positions)

        time_elapsed = time.time() - start
        global print_timing
        if print_timing:
            print('Forward pass, time: %.5f s' % time_elapsed)

        # Save context for backward
        ctx.shapes = shapes
        ctx.shape_groups = shape_groups
        ctx.canvas_width = canvas_width
        ctx.canvas_height = canvas_height
        ctx.width = width
        ctx.height = height
        ctx.num_samples_x = num_samples_x
        ctx.num_samples_y = num_samples_y
        ctx.seed = seed
        ctx.output_type = output_type
        ctx.use_prefiltering = use_prefiltering
        ctx.filter_type = ft
        ctx.filter_radius = fr
        ctx.background_image = background_image
        ctx.device = device
        ctx.num_args = len(args)
        ctx.eval_positions = eval_positions
        # Save input tensors for backward
        ctx.tensor_args = tensor_args

        return rendered_image

    @staticmethod
    def backward(ctx, grad_img):
        if not grad_img.is_contiguous():
            grad_img = grad_img.contiguous()

        device = ctx.device
        old_shapes = ctx.shapes
        old_shape_groups = ctx.shape_groups
        width = ctx.width
        height = ctx.height

        start = time.time()

        # Create fresh detached leaf tensors for all differentiable parameters.
        # We build parallel structures: new shapes/groups for the recomputed forward,
        # and a mapping from each leaf tensor to its position in the output gradient tuple.

        # grad_map: list of (arg_index, leaf_tensor) pairs
        # We'll collect all leaf tensors and use torch.autograd.grad at the end.
        leaf_tensors = []
        arg_index_for_leaf = []  # which output arg position each leaf corresponds to

        def make_leaf(t):
            """Detach and create a fresh leaf tensor with requires_grad."""
            leaf = t.detach().clone().requires_grad_(True)
            return leaf

        # Rebuild shapes with fresh leaf tensors
        new_shapes = []
        # We need to track arg positions. The args layout after the 7 header items is:
        # For each shape: shape_type, [shape params...], stroke_width
        # For each group: shape_ids, fill_color_type, [fill params...], stroke_color_type, [stroke params...], use_even_odd_rule, shape_to_canvas
        # Then: filter_type, filter_radius
        arg_idx = 7  # Start after header (canvas_w, canvas_h, num_shapes, num_shape_groups, output_type, use_prefiltering, eval_positions)

        for shape_type, shape_data in old_shapes:
            arg_idx += 1  # shape_type
            new_data = {}

            if shape_type == torch_render.ShapeType.circle:
                r = make_leaf(shape_data['radius'])
                c = make_leaf(shape_data['center'])
                leaf_tensors.extend([r, c])
                arg_index_for_leaf.extend([arg_idx, arg_idx + 1])
                new_data = {'center': c, 'radius': r}
                arg_idx += 2
            elif shape_type == torch_render.ShapeType.ellipse:
                r = make_leaf(shape_data['radius'])
                c = make_leaf(shape_data['center'])
                leaf_tensors.extend([r, c])
                arg_index_for_leaf.extend([arg_idx, arg_idx + 1])
                new_data = {'center': c, 'radius': r}
                arg_idx += 2
            elif shape_type == torch_render.ShapeType.path:
                arg_idx += 1  # num_control_points
                p = make_leaf(shape_data['points'])
                leaf_tensors.append(p)
                arg_index_for_leaf.append(arg_idx)
                new_data = {
                    'points': p,
                    'num_control_points': shape_data['num_control_points'],
                    'is_closed': shape_data['is_closed'],
                    'use_distance_approx': shape_data.get('use_distance_approx', False),
                }
                arg_idx += 1  # points
                thickness = shape_data.get('thickness')
                if thickness is not None:
                    th = make_leaf(thickness)
                    leaf_tensors.append(th)
                    arg_index_for_leaf.append(arg_idx)
                    new_data['thickness'] = th
                arg_idx += 1  # thickness (or None)
                arg_idx += 1  # is_closed
                arg_idx += 1  # use_distance_approx
            elif shape_type == torch_render.ShapeType.rect:
                pm = make_leaf(shape_data['p_min'])
                px = make_leaf(shape_data['p_max'])
                leaf_tensors.extend([pm, px])
                arg_index_for_leaf.extend([arg_idx, arg_idx + 1])
                new_data = {'p_min': pm, 'p_max': px}
                arg_idx += 2

            sw = make_leaf(shape_data['stroke_width'])
            leaf_tensors.append(sw)
            arg_index_for_leaf.append(arg_idx)
            new_data['stroke_width'] = sw
            arg_idx += 1
            new_shapes.append((shape_type, new_data))

        # Rebuild shape groups with fresh leaf tensors
        new_shape_groups = []
        for sg in old_shape_groups:
            new_sg = {
                'shape_ids': sg['shape_ids'],
                'use_even_odd_rule': sg['use_even_odd_rule'],
            }
            arg_idx += 1  # shape_ids

            # Fill color
            fct = sg.get('fill_color_type')
            arg_idx += 1  # fill_color_type
            if fct == torch_render.ColorType.constant:
                fc = make_leaf(sg['fill_color_data'])
                leaf_tensors.append(fc)
                arg_index_for_leaf.append(arg_idx)
                new_sg['fill_color_type'] = fct
                new_sg['fill_color_data'] = fc
                arg_idx += 1
            elif fct == torch_render.ColorType.linear_gradient:
                fcd = sg['fill_color_data']
                b = make_leaf(fcd['begin']); e = make_leaf(fcd['end'])
                o = make_leaf(fcd['offsets']); sc = make_leaf(fcd['stop_colors'])
                leaf_tensors.extend([b, e, o, sc])
                arg_index_for_leaf.extend([arg_idx, arg_idx+1, arg_idx+2, arg_idx+3])
                new_sg['fill_color_type'] = fct
                new_sg['fill_color_data'] = {'begin': b, 'end': e, 'offsets': o, 'stop_colors': sc}
                arg_idx += 4
            elif fct == torch_render.ColorType.radial_gradient:
                fcd = sg['fill_color_data']
                ct = make_leaf(fcd['center']); rad = make_leaf(fcd['radius'])
                o = make_leaf(fcd['offsets']); sc = make_leaf(fcd['stop_colors'])
                leaf_tensors.extend([ct, rad, o, sc])
                arg_index_for_leaf.extend([arg_idx, arg_idx+1, arg_idx+2, arg_idx+3])
                new_sg['fill_color_type'] = fct
                new_sg['fill_color_data'] = {'center': ct, 'radius': rad, 'offsets': o, 'stop_colors': sc}
                arg_idx += 4
            else:
                new_sg['fill_color_type'] = None
                new_sg['fill_color_data'] = None

            # Stroke color
            sct = sg.get('stroke_color_type')
            arg_idx += 1  # stroke_color_type
            if sct == torch_render.ColorType.constant:
                sc_tensor = make_leaf(sg['stroke_color_data'])
                leaf_tensors.append(sc_tensor)
                arg_index_for_leaf.append(arg_idx)
                new_sg['stroke_color_type'] = sct
                new_sg['stroke_color_data'] = sc_tensor
                arg_idx += 1
            elif sct == torch_render.ColorType.linear_gradient:
                scd = sg['stroke_color_data']
                b = make_leaf(scd['begin']); e = make_leaf(scd['end'])
                o = make_leaf(scd['offsets']); sc = make_leaf(scd['stop_colors'])
                leaf_tensors.extend([b, e, o, sc])
                arg_index_for_leaf.extend([arg_idx, arg_idx+1, arg_idx+2, arg_idx+3])
                new_sg['stroke_color_type'] = sct
                new_sg['stroke_color_data'] = {'begin': b, 'end': e, 'offsets': o, 'stop_colors': sc}
                arg_idx += 4
            elif sct == torch_render.ColorType.radial_gradient:
                scd = sg['stroke_color_data']
                ct = make_leaf(scd['center']); rad = make_leaf(scd['radius'])
                o = make_leaf(scd['offsets']); sc = make_leaf(scd['stop_colors'])
                leaf_tensors.extend([ct, rad, o, sc])
                arg_index_for_leaf.extend([arg_idx, arg_idx+1, arg_idx+2, arg_idx+3])
                new_sg['stroke_color_type'] = sct
                new_sg['stroke_color_data'] = {'center': ct, 'radius': rad, 'offsets': o, 'stop_colors': sc}
                arg_idx += 4
            else:
                new_sg['stroke_color_type'] = None
                new_sg['stroke_color_data'] = None

            arg_idx += 1  # use_even_odd_rule
            s2c = make_leaf(sg['shape_to_canvas'])
            leaf_tensors.append(s2c)
            arg_index_for_leaf.append(arg_idx)
            new_sg['shape_to_canvas'] = s2c
            arg_idx += 1
            new_shape_groups.append(new_sg)

        # Recompute forward with fresh leaf tensors (new autograd graph)
        with torch.enable_grad():
            rendered = torch_render.render_scene(
                width, height,
                ctx.num_samples_x, ctx.num_samples_y,
                ctx.seed,
                ctx.canvas_width, ctx.canvas_height,
                new_shapes, new_shape_groups,
                ctx.filter_type, ctx.filter_radius,
                background_image=ctx.background_image,
                use_prefiltering=True,
                device=device)

            # Compute vector-Jacobian product
            loss = (rendered * grad_img.detach()).sum()

            # Get gradients for all leaf tensors at once
            grads = torch.autograd.grad(loss, leaf_tensors,
                                        allow_unused=True)

        time_elapsed = time.time() - start
        global print_timing
        if print_timing:
            print('Backward pass, time: %.5f s' % time_elapsed)

        # Build the output gradient tuple (must match args layout exactly)
        total_args = ctx.num_args
        d_args = [None] * (6 + total_args)  # 6 for width, height, num_samples_x/y, seed, background

        # Place computed gradients at the correct positions
        for leaf, grad, ai in zip(leaf_tensors, grads, arg_index_for_leaf):
            pos = 6 + ai  # offset by the 6 non-arg params
            if grad is not None:
                d_args[pos] = grad.cpu()
            else:
                d_args[pos] = torch.zeros_like(leaf).cpu()

        return tuple(d_args)

    @staticmethod
    def render_grad(grad_img, width, height, num_samples_x, num_samples_y,
                    seed, background_image, *args):
        """Compute translation gradient image (for compatibility)."""
        return torch.zeros(height, width, 2, device=pydiffvg.get_device())


def _render_sdf(width, height, num_samples_x, num_samples_y, seed,
                canvas_width, canvas_height, shapes, shape_groups, device,
                eval_positions):
    """Render signed distance field."""
    if eval_positions is not None and eval_positions.shape[0] > 0:
        pts = eval_positions.to(device)
        sdf = torch.zeros(pts.shape[0], 1, device=device)
    else:
        py_coords = torch.arange(height, device=device, dtype=torch.float32)
        px_coords = torch.arange(width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(py_coords, px_coords, indexing='ij')
        pts = torch.stack([grid_x.reshape(-1) + 0.5, grid_y.reshape(-1) + 0.5], dim=-1)
        pts[:, 0] *= canvas_width / width
        pts[:, 1] *= canvas_height / height
        sdf = torch.zeros(height, width, 1, device=device)

    # Compute distance to nearest shape
    min_dist = torch.full((pts.shape[0],), float('inf'), device=device)
    min_inside = torch.zeros(pts.shape[0], device=device, dtype=torch.bool)

    for sg in shape_groups:
        shape_to_canvas = sg['shape_to_canvas']
        canvas_to_shape = torch.linalg.inv(shape_to_canvas)
        local_pts = torch_render.xform_pt(canvas_to_shape, pts)
        use_even_odd = sg.get('use_even_odd_rule', True)

        for shape_id in sg['shape_ids']:
            sid = shape_id.item() if isinstance(shape_id, torch.Tensor) else shape_id
            shape_type, shape_data = shapes[sid]
            _, dist, found = torch_render.compute_distance(shape_type, shape_data, local_pts)
            better = found & (dist < min_dist)
            min_dist = torch.where(better, dist, min_dist)

            # Check inside
            wn = torch_render.compute_winding_number(shape_type, shape_data, local_pts)
            if use_even_odd:
                inside = wn.abs() % 2 == 1
            else:
                inside = wn != 0
            min_inside = torch.where(better, inside, min_inside)

    signed_dist = torch.where(min_inside, -min_dist, min_dist)

    if eval_positions is not None and eval_positions.shape[0] > 0:
        return signed_dist.unsqueeze(-1)
    else:
        return signed_dist.reshape(height, width, 1)


def _convert_filter_type(ft):
    """Convert C++ FilterType enum or our FilterType to torch_render.FilterType."""
    if isinstance(ft, torch_render.FilterType):
        return ft
    # Handle the C++ diffvg.FilterType enum values
    try:
        import diffvg
        if ft == diffvg.FilterType.box:
            return torch_render.FilterType.box
        elif ft == diffvg.FilterType.tent:
            return torch_render.FilterType.tent
        elif ft == diffvg.FilterType.parabolic:
            return torch_render.FilterType.parabolic
        elif ft == diffvg.FilterType.hann:
            return torch_render.FilterType.hann
    except (ImportError, AttributeError):
        pass
    # Default mappings by integer value
    mapping = {0: torch_render.FilterType.box,
               1: torch_render.FilterType.tent,
               2: torch_render.FilterType.parabolic,
               3: torch_render.FilterType.hann}
    return mapping.get(int(ft), torch_render.FilterType.box)
