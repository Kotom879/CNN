import java.util.Random;

public class ConvLayer {
	double[][][] input;
	double[][][] output;
	
	double[][][][] kernel;
	double[][][] bias;
	
	double[][][][] kernelGradient;
	double[][][] biasGradient;
	
	ConvLayer(int depth, int inputSize, int numberOfKernels,  int kernelSize){
		this.input = new double[depth][inputSize][inputSize];
		this.output = new double[numberOfKernels][inputSize - kernelSize + 1][inputSize - kernelSize + 1];
		
		this.kernel = new double[numberOfKernels][depth][kernelSize][kernelSize];
		this.bias = new double[numberOfKernels][inputSize - kernelSize + 1][inputSize - kernelSize + 1];
		
		this.kernelGradient = new double[numberOfKernels][depth][kernelSize][kernelSize];
		this.biasGradient = new double[numberOfKernels][inputSize - kernelSize + 1][inputSize - kernelSize + 1];
	}
	void randomize() {
		Random rand = new Random();
		for(int i = 0; i < this.kernel.length; i++) {
			for(int j = 0; j < this.bias[i].length; j++) {
				for(int k = 0; k < this.bias[i][j].length; k++) {
					this.bias[i][j][k] = rand.nextDouble() - 0.5;
					if(this.bias[i][j][k] == 0) bias[i][j][k] += 0.1;
				}
			}
			for(int j = 0; j < this.kernel[i].length; j++) {
				for(int k = 0; k < this.kernel[i][j].length; k++) {
					for(int l = 0; l < this.kernel[i][j][k].length; l++) {
						this.kernel[i][j][k][l] = rand.nextDouble() - 0.5;
					}
				}
			}
		}
	}//end of randomize() method
	//main methods
	double[][][] forward(double[][][] input){
		this.input = input;
		this.output = new double[output.length]
				[output[0].length][output[0][0].length];
		for(int i = 0; i < output.length; i++) {
			for(int j = 0; j < output[i].length; j++) {
				for(int k = 0; k < output[i][j].length; k++) {
					output[i][j][k] = crossCorrelate(i, j, k);
				}
			}
		}
		return output;
	}//end of forward() method
	double[][][] backward(double[][][] error){
		double[][][] inputGradient = new double[input.length]
				[input[0].length][input[0][0].length];
		for(int i = 0; i < kernel.length; i++) {
			biasGradient[i] = matrixAdd(biasGradient[i], error[i]);
			for(int j = 0; j < kernel[i].length; j++) {
				kernelGradient[i][j] = matrixAdd(kernelGradient[i][j], crossCorrelate(input[j], error[i]));
			}
		}
		for(int i = 0; i < kernel.length; i++) {
			for(int j = 0; j < kernel[i].length; j++) {
				inputGradient[j] = matrixAdd(inputGradient[j], fullConvolution(error[i], kernel[i][j]));
			}
		}
		return inputGradient;
	}//end of backward() method
	void updateConfig(double learningRate) {
		for(int i = 0; i < kernel.length; i++) {//number of kernels
			//bias
			for(int j = 0; j < bias[i].length; j++) {
				for(int k = 0; k < bias[i][j].length; k++) {
					bias[i][j][k] -= biasGradient[i][j][k] * learningRate; 
				}
			}
			//kernels
			for(int j = 0; j < kernel[i].length; j++) {//depth
				for(int k = 0; k < kernel[i][j].length; k++) {//kernelSize
					for(int l = 0; l < kernel[i][j][k].length; l++) {//kernelSize
						kernel[i][j][k][l] -= kernelGradient[i][j][k][l] * learningRate;
					}
				}
			}
		}
		kernelGradient = new double[kernel.length][kernel[0].length][kernel[0][0].length][kernel[0][0][0].length];
		biasGradient = new double[bias.length][bias[0].length][bias[0][0].length];
	}//end of updateConfig() method
	
	//maths
	double crossCorrelate(int outputNumber, int x, int y) {
		double output = 0;
		for(int i = 0; i < input.length; i++) {//depth
			for(int j = 0; j < kernel[0][0][0].length; j++) {//kernelSize
				for(int k = 0; k < kernel[0][0][0].length; k++) {//kernelSize
					output += input[i][j + x][k + y] 
							* kernel[outputNumber][i][j][k];
				}
			}
		}
		output += bias[outputNumber][x][y];
		return output;
	}//end of crossCorrelate() method
	double[][] matrixAdd(double[][] a, double[][] b){
		if(a.length != b.length || a[0].length != b[0].length) {
			System.out.print("\nmatrixAdd: matrices of unequal dimensions");
			return new double[0][0];
		}
		double[][] c = new double[a.length][a[0].length]; 
		for(int i = 0; i < a.length; i++) {
			for(int j = 0; j < a[i].length; j++) {
				c[i][j] = a[i][j] + b[i][j];
			}
		}
		return c;
	}//end of matrixAdd() method
	double[][] crossCorrelate(double[][] a, double[][] b){
		int width = a.length - b.length + 1;
		int length = a[0].length - b[0].length + 1;
		double[][] output = new double[width][length];
		for(int i = 0; i < width; i++) {
			for(int j = 0; j < length; j++) {
				output[i][j] = weightedSum(i, j, a, b);
			}
		}
		return output;
	}//end of crossCorrelate()method
	double weightedSum(int x, int y, double[][] a, double[][] b){
		double sum = 0;
		for(int i = 0; i < b.length; i++) {
			for(int j = 0; j < b[i].length; j++) {
				sum += a[x + i][y + j] * b[i][j];
			}
		}
		return sum;
	}//end of weightedSum() method
	double[][] fullConvolution(double[][] a, double[][] b) {
		double[][] out = fullCrossCorrelate(a, rot180(b));
		return out;
	}//end of fullConvolution() method
	double[][] fullCrossCorrelate(double[][] a, double[][] b){
		int outwidth = a.length + b.length - 1;
		int outlength = a[0].length + b[0].length - 1;
		double[][] output = new double[outwidth][outlength];
		
		int inwidth = a.length + 2 * (b.length - 1);
		int inlength = a[0].length + 2 * (b[0].length - 1);
		double[][] bigInput = new double[inwidth][inlength];
	
		for(int i = b.length - 1; i < a.length + b.length - 1; i++){
			for(int j = b[0].length - 1; j < a[0].length + b[0].length - 1; j++){
				bigInput[i][j] = a[i - b.length + 1][j - b[0].length + 1];
			}
		}
		
		for(int i = 0; i < outwidth; i++) {
			for(int j = 0; j < outlength; j++) {
				output[i][j] = weightedSum(i, j, bigInput, b);
			}
		}
		return output;
	}//end of fullCrossCorrelate() method
	double[][] rot180(double[][] a){
		double[][] output = new double[a.length][a[0].length];
		for(int i = 0; i < a.length; i++){
			for(int j = 0; j < a[i].length; j++){
				output[a.length -i - 1][a[i].length - j - 1] = a[i][j];
			}
		}
		return output;
	}//end of rot180() method
}//end of ConvLayer class
