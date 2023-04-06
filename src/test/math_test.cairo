use array::ArrayTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::int33;
use onnx_cairo::operators::math::int33::i33;
use linear_regression::onnx_cairo::operators::math::matrix::MatrixTrait;
use linear_regression::onnx_cairo::operators::math::matrix::Matrix;
use linear_regression::lr::linear_regression::LRTrait;


use linear_regression::math;

#[test]
fn test_math() {
    assert(math::add(2, 3) == 5, 'invalid');
    // assert(math::fib(0, 1, 10) == 55, 'invalid');
}

#[test]
#[available_gas(99999999999999999)]
fn lr_test() -> Matrix {
    // Generate some random input data and their corresponding labels
    // For X Matrix
    let mut arr = ArrayTrait::<i33>::new();
    
    let val_0 = i33 { inner: 0_u32, sign: false };
    let val_1 = i33 { inner: 1_u32, sign: false };
    let val_2 = i33 { inner: 2_u32, sign: false };
    let val_3 = i33 { inner: 3_u32, sign: false };
    let val_4 = i33 { inner: 4_u32, sign: false };
    let val_5 = i33 { inner: 1_u32, sign: false };
    let val_6 = i33 { inner: 2_u32, sign: false };
    let val_7 = i33 { inner: 3_u32, sign: false };
    let val_8 = i33 { inner: 4_u32, sign: false };
    let val_9 = i33 { inner: 5_u32, sign: false };

    arr.append(val_0);
    arr.append(val_1);
    arr.append(val_2);
    arr.append(val_3);
    arr.append(val_4);
    arr.append(val_5);
    arr.append(val_6);
    arr.append(val_7);
    arr.append(val_8);
    arr.append(val_9);

    // For Y_true values
    let mut arr_y = ArrayTrait::<i33>::new();
    
    let val_y_0 = i33 { inner: 0_u32, sign: false };
    let val_y_1 = i33 { inner: 1_u32, sign: false };
    let val_y_2 = i33 { inner: 2_u32, sign: false };
    let val_y_3 = i33 { inner: 3_u32, sign: false };
    let val_y_4 = i33 { inner: 4_u32, sign: false };

    arr_y.append(val_y_0);
    arr_y.append(val_y_1);
    arr_y.append(val_y_2);
    arr_y.append(val_y_3);
    arr_y.append(val_y_4);

    // For W_obvs values
    let mut arr_W = ArrayTrait::<i33>::new();
    let val_W_0 = i33 { inner: 2_u32, sign: false };
    let val_W_1 = i33 { inner: 1_u32, sign: false };
    arr_W.append(val_W_0);
    arr_W.append(val_W_1);

    // For bais values
    let mut arr_b = ArrayTrait::<i33>::new();
    let val_b = i33 { inner: 1_u32, sign: false };
    arr_b.append(val_b);

    // Data 
    let X = MatrixTrait::new(5_u32, 2_u32, arr);
    let y = MatrixTrait::new(5_u32, 1_u32, arr_y);

    // Create a linear regression model
    let W = MatrixTrait::new(1_u32, 2_u32, arr_W);
    let b = MatrixTrait::new(1_u32, 1_u32, arr_b);
    let lr = LRTrait::new(W, b);

    // Predict the labels for the input data using the linear regression model
    let y_pred = lr.predict(@X);

    // Compare the predicted labels with the actual labels
    let epsilon = 0; // Set a small epsilon value
    //assert((y_pred.get(1_usize, 1_usize) - y.get(1_usize, 1_usize)) < epsilon, 'distance less than epsilon');
    y_pred
    
}