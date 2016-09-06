package com.github.zjiajun.regression.logistic;

import com.github.zjiajun.regression.costfunction.CostFunctionMinimizer;
import org.jblas.DoubleMatrix;

/**
 * @author zhujiajun
 * @since 16/9/6
 */
public class LogisticGradientDescentCostFunctionMinimizer implements CostFunctionMinimizer {

    private double alpha;
    private double iterations;

    /**
     * Constructor.
     *
     * @param alpha Alpha parameter which configures how small each grandient descent is.
     * @param iterations Number of iterations for algorithm.
     */
    public LogisticGradientDescentCostFunctionMinimizer(double alpha, double iterations, boolean normalizeFeatures) {
        super();
        this.alpha = alpha;
        this.iterations = iterations;
    }


    public DoubleMatrix minimizeCostFunction(DoubleMatrix features, DoubleMatrix values) {
        int size = values.rows;
        DoubleMatrix theta = DoubleMatrix.zeros(features.columns, 1);
        for (int i = 0; i < iterations; i++) {
            DoubleMatrix featuresTranspose = features.transpose();
            DoubleMatrix hx = new LogaritmicCostFunction().sigmoid(features.mmul(theta));
            DoubleMatrix delta = (featuresTranspose.mmul(hx).sub(featuresTranspose.mmul(values))).div(size);
            // theta = theta - (1 / (m)) * (X' * hx - X' * y)
            theta = theta.sub(delta.mul(alpha));
        }
        return theta;
    }

}
