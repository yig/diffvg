"""Minimal test for the pure PyTorch diffvg renderer."""
import torch
import pydiffvg


def run_tests(device_name):
    dev = torch.device(device_name)
    pydiffvg.set_device(dev)
    print(f"\n{'='*60}")
    print(f"Testing on device: {device_name}")
    print(f"{'='*60}")

    print("\n--- Test 1: Forward pass (single circle) ---")
    canvas_width = 64
    canvas_height = 64
    circle = pydiffvg.Circle(radius=torch.tensor(20.0),
                             center=torch.tensor([32.0, 32.0]))
    shapes = [circle]
    circle_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0]))
    shape_groups = [circle_group]

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(64, 64, 2, 2, 0, None, *scene_args)

    print(f"  Image shape: {img.shape}")
    print(f"  Image range: [{img.min().item():.4f}, {img.max().item():.4f}]")
    print(f"  Center pixel (32,32): {[round(x, 4) for x in img[32, 32].tolist()]}")
    assert img.shape == (64, 64, 4)
    assert img[32, 32, 1].item() > 0.3
    assert img[0, 0, 3].item() < 0.1
    print("  PASSED!")

    target = img.clone().detach()

    print("\n--- Test 2: Backward pass (gradient computation) ---")
    radius_n = torch.tensor(15.0 / 64.0, requires_grad=True)
    center_n = torch.tensor([28.0 / 64.0, 36.0 / 64.0], requires_grad=True)
    color = torch.tensor([0.3, 0.2, 0.8, 1.0], requires_grad=True)

    circle.radius = radius_n * 64
    circle.center = center_n * 64
    circle_group.fill_color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img2 = render(64, 64, 2, 2, 1, None, *scene_args)

    loss = (img2 - target).pow(2).sum()
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()
    print(f"  radius_n.grad: {radius_n.grad}")
    print(f"  center_n.grad: {center_n.grad}")
    print(f"  color.grad: {color.grad}")
    assert radius_n.grad is not None and radius_n.grad.abs().item() > 0
    assert center_n.grad is not None
    assert color.grad is not None
    print("  PASSED!")

    print("\n--- Test 3: Optimization (5 iterations) ---")
    radius_n = torch.tensor(15.0 / 64.0, requires_grad=True)
    center_n = torch.tensor([28.0 / 64.0, 36.0 / 64.0], requires_grad=True)
    color = torch.tensor([0.3, 0.2, 0.8, 1.0], requires_grad=True)
    optimizer = torch.optim.Adam([radius_n, center_n, color], lr=1e-2)

    losses = []
    for t in range(5):
        optimizer.zero_grad()
        circle.radius = radius_n * 64
        circle.center = center_n * 64
        circle_group.fill_color = color
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        img_opt = render(64, 64, 2, 2, t + 1, None, *scene_args)
        loss = (img_opt - target).pow(2).sum()
        losses.append(loss.item())
        print(f"  iter {t}: loss={loss.item():.4f}")
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print("  PASSED!")

    print(f"\nAll tests passed on {device_name}!")


run_tests('cpu')

if torch.cuda.is_available():
    run_tests('cuda')
else:
    print("\nCUDA not available, skipping.")

if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    run_tests('mps')
else:
    print("\nMPS not available, skipping.")
