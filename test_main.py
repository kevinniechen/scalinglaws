import pytest
import math
from main import upstream_loss, downstream_error, gradient_descent_multiplicative, format_tokens

def test_upstream_loss():
    # Test with default parameters
    assert math.isclose(upstream_loss(1e20, 4e10), 2.3400, rel_tol=1e-4)
    
    # Test with custom parameters
    assert math.isclose(upstream_loss(1e20, 4e10, E=2, alpha=200, beta=350, eta=0.14), 2.4956, rel_tol=1e-4)

def test_downstream_error():
    # Test with default parameters
    assert math.isclose(downstream_error(2.34), 0.1570, rel_tol=1e-4)
    
    # Test with custom parameters
    assert math.isclose(downstream_error(2.34, epsilon=0.9, k=2.3, gamma=0.7), 0.1899, rel_tol=1e-4)

def test_gradient_descent_multiplicative():
    # Test optimization for a small scale problem
    optimized_D = gradient_descent_multiplicative(1e18, 1e9, factor=1.1, num_iterations=100)
    assert 1e8 < optimized_D < 1e10  # Rough check that the result is reasonable

def test_format_tokens():
    assert format_tokens(1e6) == "1.0M"
    assert format_tokens(1e9) == "1.0B"
    assert format_tokens(1e12) == "1.0T"
    assert format_tokens(1.5e11) == "150.0M"

def test_scaling_robustness():
    # Test upstream_loss scaling
    base_loss = upstream_loss(1e20, 4e10)
    scaled_loss = upstream_loss(1e21, 4e11)
    assert math.isclose(base_loss, scaled_loss, rel_tol=1e-2)

    # Test downstream_error scaling
    base_error = downstream_error(base_loss)
    scaled_error = downstream_error(scaled_loss)
    assert math.isclose(base_error, scaled_error, rel_tol=1e-2)

    # Test gradient_descent_multiplicative scaling
    base_D = gradient_descent_multiplicative(1e20, 4e10, num_iterations=100)
    scaled_D = gradient_descent_multiplicative(1e21, 4e11, num_iterations=100)
    assert math.isclose(base_D / 1e20, scaled_D / 1e21, rel_tol=1e-1)

if __name__ == "__main__":
    pytest.main()