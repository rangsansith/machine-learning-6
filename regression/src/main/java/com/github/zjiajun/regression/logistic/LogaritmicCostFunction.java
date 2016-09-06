package com.github.zjiajun.regression.logistic;

import com.github.zjiajun.regression.costfunction.CostFunction;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * @author zhujiajun
 * @since 16/9/6
 */
public class LogaritmicCostFunction implements CostFunction {

    public double computeCost(DoubleMatrix features, DoubleMatrix values, DoubleMatrix theta) {
        int size = values.rows;
        DoubleMatrix ones = DoubleMatrix.ones(values.columns, values.rows);
        // -1 .* y .* log(sigmoid(X*theta))
        DoubleMatrix first = values.mul(-1.0).mul(MatrixFunctions.log(sigmoid(features.mmul(theta))));
        // (1 - y) .* (log(1 - sigmoid(X*theta)))
        DoubleMatrix second = ones.sub(values).mul(MatrixFunctions.log(ones.sub(sigmoid(features.mmul(theta)))));
        return (first.sub(second)).sum() / size;
    }

    /**
     * Sigmoid function. Formula: g = 1 ./ (1 + (exp(-1 .* z)));
     *
     * @param z Matrix, for which elements sigmoid function is calculated.
     * @return Matrix with elements from sigmoid function.
     */
    public DoubleMatrix sigmoid(DoubleMatrix z) {
        DoubleMatrix ones = DoubleMatrix.ones(z.rows, z.columns);
        return ones.div(ones.add(MatrixFunctions.exp(z.mul(-1.0))));
    }

}
