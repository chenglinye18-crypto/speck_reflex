import pytest
import torch


@pytest.mark.gpu
def test_cuda_matrix_and_inference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(17)
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    result = a @ b
    model = torch.nn.Linear(256, 10, bias=False).cuda()
    output = model(result)
    torch.cuda.synchronize()
    assert output.is_cuda and output.shape == (256, 10)
    assert torch.isfinite(output).all()
