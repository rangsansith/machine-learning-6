package com.github.zjiajun.regression.costfunction;

import org.jblas.DoubleMatrix;

/**
 * @author zhujiajun
 * @since 16/9/6
 */
public interface CostFunction {

    /**
     * @param features Features (or x1, x2, x3 ...  values).
     * @param values Function values (or y).
     * @param theta Theta values vector.
     *
     * @return Cost of prediction.
     */
    double computeCost(DoubleMatrix features, DoubleMatrix values, DoubleMatrix theta);
}
