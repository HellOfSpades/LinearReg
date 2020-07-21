package com.miron.main;

import org.apache.commons.math.linear.MatrixUtils;
import org.apache.commons.math.linear.RealMatrix;

import java.util.Scanner;

public class Main {
    //final variables
    static final int length = 300;
    static final int parametersNum = 1;
    static final double a = 0.0000000006;
    static final int learningLoops = 2000;

    //the arrays
    static double[][] X = new double[length][parametersNum];
    static double[][] Y = new double[length][1];
    static double[][] Theta = new double[1][parametersNum+1];

    //Matrices
    static RealMatrix mX;
    static RealMatrix mY;
    static RealMatrix mTheta;


    public static void main(String[] args){
        makeData();
        double[][] newX = addOnes(X);
        Theta = setRandom(Theta,4);
        //setting up the matrices
        mX = MatrixUtils.createRealMatrix(newX);
        mY = MatrixUtils.createRealMatrix(Y);
        mTheta = MatrixUtils.createRealMatrix(Theta);

        //Make it so the neural network learns
        learn();

        //test out that it works by imputing costum data
        Scanner scanner = new Scanner(System.in);
        while(true){

            String[] s = scanner.nextLine().split(" ");
            double[][] customData = new double[1][s.length+1];
            customData[0][0] = 1;
            for(int i = 1;i<=s.length;i++){
                customData[0][i] = Double.parseDouble(s[i-1]);
            }
            printMatrix(use(MatrixUtils.createRealMatrix((customData))));
        }
    }

    //loops a learningLoops amount of times
    //makes a prediction using the use() method
    //it then changes the Theta matrix so it goes down the cost function gradient
    public static void learn() {
        for (int i = 0; i < learningLoops; i++) {

            RealMatrix h = use(mX);
            RealMatrix subtracted = h.subtract(mY);
            //printing the cost to make shure it decreases
            System.out.println("Cost at loop "+i);
            System.out.println(cumputeCost(subtracted));
            System.out.print("Theta is");
            printMatrix(mTheta);
            System.out.println("-----------------------------------------------");

            RealMatrix newTheta = MatrixUtils.createRealMatrix(1,parametersNum+1);



            //updating Theta
            for(int n = 0;n<=parametersNum;n++){

                //this is done to easier get the coulumn vector

                RealMatrix XCoulumnMatrix = mX.getColumnMatrix(n);
                RealMatrix MoltipliedMatrices = subtracted.multiply(XCoulumnMatrix.transpose());
                double sum = 0;
                for(int m = 0;m<MoltipliedMatrices.getColumnDimension();m++){
                    for(int k = 0;k<MoltipliedMatrices.getRowDimension();k++){
                        sum+=MoltipliedMatrices.getEntry(k,m);
                    }

                }

                newTheta.setEntry(0,n,mTheta.getEntry(0,n)-a*sum);
            }
            //changing the old theta to be the newTheta
            mTheta = newTheta;

        }
    }

    //returns the cost of data X and Y with Theta weights
    public static double cumputeCost(RealMatrix h){

        double sum = 0;
        for(int i = 0;i<length;i++){
            sum+=Math.pow(h.getEntry(i,0),2);
        }

        return sum/(2*length);
    }

    //prints RealMatrix matrix to the console
    public static void printMatrix(RealMatrix matrix){
        System.out.println("");
        for(int i = 0;i<matrix.getRowDimension();i++){
            for(int n = 0;n<matrix.getColumnDimension();n++){
                System.out.print(matrix.getEntry(i,n)+"\t");
            }
            System.out.println("");
        }
    }

    //returns a prediction based on the input X;
    public static RealMatrix use(RealMatrix X){

        return X.multiply(mTheta.transpose());
    }
    //adds a 1 to the begining of each piece of data
    public static double[][] addOnes(double[][] X){
        double[][] newX = new double[length][parametersNum+1];

        for(int i = 0;i<length;i++){
            newX[i][0] = 1;
            for(int n = 0;n<parametersNum;n++){
                newX[i][n+1] = X[i][n];
            }
        }

        return newX;
    }

    //sets random values for the target 2 dimensional array with numbers from -max to max
    public static double[][] setRandom(double[][] array, int max){

        for(int i = 0;i<array.length;i++){
            for(int n = 0;n<array[i].length;n++){
                array[i][n] = Math.random()*max*2-max;
            }
        }

        return array;
    }
    //method that makes random numbers for X and assighns the apropriate Y using the formula method
    public static void makeData(){
        for(int i = 0;i<length;i++){
            for(int n = 0;n<parametersNum;n++){
                X[i][n] = Math.random()*length;
            }
            Y[i] = formula(X[i]);

        }
    }
    //formula for the equation
    public static double[] formula(double[] X){
        double[] Y = new double[1];
        Y[0] = 2+X[0]*2;
        return Y;
    }
}
