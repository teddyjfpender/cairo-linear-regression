use array::ArrayTrait;
use option::OptionTrait;

use linear_regression::onnx_cairo::operators::math::int33;
use linear_regression::onnx_cairo::operators::math::int33::i33;
use linear_regression::onnx_cairo::operators::math::matrix::Matrix;
use linear_regression::onnx_cairo::operators::math::matrix::MatrixTrait;


#[derive(Drop)]
struct LinearRegression {
    W: Matrix,
    b: Matrix,
}

trait LRTrait {
    fn new(W: Matrix, b: Matrix) -> LinearRegression;
    fn forward_prop(self: @LinearRegression, X: @Matrix) -> Matrix;
    fn predict(self: @LinearRegression, X: @Matrix) -> Matrix;
}

impl LRImpl of LRTrait {
    fn new(W: Matrix, b: Matrix) -> LinearRegression {
        lr_new(W, b)
    }

    fn forward_prop(self: @LinearRegression, X: @Matrix) -> Matrix {
        let mut Z_temp = self.W.dot(X);
        let mut Z = Z_temp.add(self.b);
        Z
    }

    fn predict(self: @LinearRegression, X: @Matrix) -> Matrix {
        self.forward_prop(X)
    }
}

fn lr_new(W: Matrix, b: Matrix) -> LinearRegression {
    LinearRegression { W: W, b: b }
}
