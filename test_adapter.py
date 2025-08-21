#!/usr/bin/env python3
"""
Test script for the optimizer adapter.

This script verifies that the AdamATan2 replacement works correctly
and can be used as a drop-in replacement for the original package.
"""

import torch
import torch.nn as nn
from optimizer_adapter import AdamATan2, AdamWAdapter, create_optimizer


def test_optimizer_adapter():
    """Test the optimizer adapter functionality."""
    print("🧪 Testing Optimizer Adapter...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Test data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # Loss function
    criterion = nn.MSELoss()
    
    print("\n1. Testing AdamATan2...")
    try:
        optimizer1 = AdamATan2(model.parameters(), lr=0.001)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer1.step()
        optimizer1.zero_grad()
        
        print("   ✅ AdamATan2 works correctly")
        
    except Exception as e:
        print(f"   ❌ AdamATan2 failed: {e}")
        return False
    
    print("\n2. Testing AdamWAdapter...")
    try:
        optimizer2 = AdamWAdapter(model.parameters(), lr=0.001)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer2.step()
        optimizer2.zero_grad()
        
        print("   ✅ AdamWAdapter works correctly")
        
    except Exception as e:
        print(f"   ❌ AdamWAdapter failed: {e}")
        return False
    
    print("\n3. Testing create_optimizer factory...")
    try:
        optimizer3 = create_optimizer("adam_atan2", params=model.parameters(), lr=0.001)
        print("   ✅ create_optimizer with 'adam_atan2' works")
        
        optimizer4 = create_optimizer("adamw", params=model.parameters(), lr=0.001)
        print("   ✅ create_optimizer with 'adamw' works")
        
        optimizer5 = create_optimizer("adam", params=model.parameters(), lr=0.001)
        print("   ✅ create_optimizer with 'adam' works")
        
    except Exception as e:
        print(f"   ❌ create_optimizer failed: {e}")
        return False
    
    print("\n4. Testing import replacement...")
    try:
        # Simulate the original import
        from optimizer_adapter import AdamATan2 as OriginalAdamATan2
        
        # Use it as if it were the original package
        optimizer6 = OriginalAdamATan2(model.parameters(), lr=0.001)
        
        # Test a training step
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer6.step()
        optimizer6.zero_grad()
        
        print("   ✅ Import replacement works correctly")
        
    except Exception as e:
        print(f"   ❌ Import replacement failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The optimizer adapter is working correctly.")
    return True


def test_cuda_availability():
    """Test CUDA availability and PyTorch installation."""
    print("\n🔍 Testing CUDA and PyTorch...")
    
    try:
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU device: {torch.cuda.get_device_name(0)}")
            print(f"   GPU count: {torch.cuda.device_count()}")
        else:
            print("   ⚠️  CUDA not available - training will use CPU")
            
    except Exception as e:
        print(f"   ❌ Error checking CUDA: {e}")


def main():
    """Main test function."""
    print("🚀 HRM Optimizer Adapter Test Suite")
    print("=" * 40)
    
    # Test CUDA
    test_cuda_availability()
    
    # Test optimizer adapter
    success = test_optimizer_adapter()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("\nNext steps:")
        print("1. You can now use the optimizer adapter in your training code")
        print("2. Replace 'from adam_atan2 import AdamATan2' with 'from optimizer_adapter import AdamATan2'")
        print("3. The rest of your code should work unchanged")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return success


if __name__ == "__main__":
    main()
