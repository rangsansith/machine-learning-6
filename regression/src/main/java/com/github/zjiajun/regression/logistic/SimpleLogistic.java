package com.github.zjiajun.regression.logistic;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * @author zhujiajun
 * @since 16/9/6
 */
public class SimpleLogistic {
    /** the learning rate */
    private double rate;

    /** the weight to learn */
    private double[] weights;

    /** the number of iterations */
    private int ITERATIONS = 3000_00;

    public SimpleLogistic(int n) {
        this.rate = 0.0001;
        weights = new double[n];
    }

    public double computeCost(List<Instance> instances) {
        double lik = 0.0;
        for (int i=0; i<instances.size(); i++) {
            double[] x = instances.get(i).x;
            double predicted = classify(x);
            double label = instances.get(i).label;
            lik += label * Math.log(predicted) + (1 - label) * Math.log(1 - predicted);
        }
        return lik;
    }

    public void train(List<Instance> instances) {
        for (int n=0; n<ITERATIONS; n++) {
            double lik = 0.0;
            for (int i=0; i<instances.size(); i++) {
                double[] x = instances.get(i).x;
                double predicted = classify(x);
                double label = instances.get(i).label;
                for (int j=0; j<weights.length; j++) {
                    weights[j] = weights[j] + rate * (label - predicted) * x[j];
                }
                // not necessary for learning
                lik += label * Math.log(classify(x)) + (1 - label) * Math.log(1 - classify(x));
            }
            System.out.println("iteration: " + n + " " + Arrays.toString(weights) + " mle: " + lik);
        }
    }

    private double classify(double[] x) {
        double logit = .0;
        for (int i=0; i<weights.length;i++)  {
            logit += weights[i] * x[i];
        }
        return sigmoid(logit);
    }

    private static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static class Instance {
        public double label;
        public double [] x;

        public Instance(double label, double[] x) {
            this.label = label;
            this.x = x;
        }
    }

    public static List<Instance> readData(String file) throws IOException {
        List<Instance> dataset = new ArrayList<>();
        String str;
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));
            while ((str = reader.readLine()) != null) {
                String[] split = str.split(",");
                double[] data = new double[split.length];
                data[0] = 1.0;
                data[1] = Double.parseDouble(split[0]);
                data[2] = Double.parseDouble(split[1]);
                double label = Double.parseDouble(split[2]);
                Instance instance = new Instance(label, data);
                dataset.add(instance);
            }
        } finally {
            if (reader != null) reader.close();
        }
        return dataset;
    }

    public static List<Instance> readDataSet(String file) throws FileNotFoundException {
        List<Instance> dataset = new ArrayList<>();
        Scanner scanner = null;
        try {
            scanner = new Scanner(new File(file));
            while(scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (line.startsWith("#")) {
                    continue;
                }
                String[] columns = line.split("\\s+");

                // skip first column and last column is the label
                int i = 1;
                double[] data = new double[columns.length-2];
                for (i=1; i<columns.length-1; i++) {
                    data[i-1] = Double.parseDouble(columns[i]);
                }
                double label = Double.parseDouble(columns[i]);
                Instance instance = new Instance(label, data);
                dataset.add(instance);
            }
        } finally {
            if (scanner != null)
                scanner.close();
        }
        return dataset;
    }


    public static void main(String... args) throws IOException {
        String file = Thread.currentThread().getContextClassLoader().getResource("ex2data1.txt").getFile();
//        List<Instance> instances = readDataSet(file);
        List<Instance> instances = readData(file);
        SimpleLogistic logistic = new SimpleLogistic(3);
        double cost = logistic.computeCost(instances);
        System.out.println("Cost value : " + cost / instances.size());

        logistic.train(instances);

        double[] x = {1,0,34.62365962451697,78.0246928153624};
        System.out.println("prob(1|x) = " + logistic.classify(x));

        double[] x2 = {1.0,60.18259938620976,86.30855209546826};
        System.out.println("prob(1|x2) = " + logistic.classify(x2));

        double[] x3 = {1.0,82.30705337399482,76.48196330235604};
        System.out.println("prob(1|x3) = " + logistic.classify(x3));

        double[] x4 = {1.0,69.36458875970939,97.71869196188608};
        System.out.println("prob(1|x4) = " + logistic.classify(x4));

        double[] x5 = {1.0,39.53833914367223,76.03681085115882};
        System.out.println("prob(1|x5) = " + logistic.classify(x5));

    }
}
