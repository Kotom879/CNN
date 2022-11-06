import java.io.IOException;

public class Main {

	public static void main(String[] args) throws IOException {
		
		MnistMatrix[] mnistMatrix = new MnistDataReader().readData("D:\\MNIST\\train-images.idx3-ubyte", "D:\\MNIST\\train-labels.idx1-ubyte");
		MnistMatrix[] test = new MnistDataReader().readData("D:\\MNIST\\t10k-images.idx3-ubyte", "D:\\MNIST\\t10k-labels.idx1-ubyte");
		
		ConvLayer[] convLayers = new ConvLayer[2];
		PoolLayer[] poolLayers = new PoolLayer[2];
		DenseLayer[] denseLayers = new DenseLayer[2];
		ActivLayer[] activLayers = new ActivLayer[4];
		
		convLayers[0] 	= new ConvLayer	(1, 28, 8, 3);
		poolLayers[0]	= new PoolLayer	(8, 26);
		activLayers[0] 	= new LReLU		(8, 13);
		convLayers[1] 	= new ConvLayer	(8, 13, 16, 3);
		poolLayers[1]	= new PoolLayer	(16, 11);
		activLayers[1] 	= new LReLU		(16, 5);
		Reshape reshape = new Reshape	(16, 5);
		denseLayers[0] 	= new DenseLayer(400, 80);
		activLayers[2] 	= new Sigmoid	(80);
		denseLayers[1] 	= new DenseLayer(80, 10);
		activLayers[3] 	= new Sigmoid	(10);
		
		//losowanie parametrów
		for(int i = 0; i < convLayers.length; i++) {
			convLayers[i].randomize();
		}
		for(int i = 0; i < denseLayers.length; i++) {
			denseLayers[i].randomize();
		}
		
		double[][][] output3d = new double[0][0][0];
		double[] output = new double[0];
		double[][][] error3d = new double[0][0][0];
		double[] error = new double[0];	
		
		//==============================================================================
		
		int epochs = 3;
		int batchSize = 10;
		double learningRate = 0.01;
		int numberOfBatches = (60000 / batchSize) * epochs;
		System.out.print("\nLearning...\nNumber of batches: " + numberOfBatches);
		for(int i = 0; i < numberOfBatches; i++) {
			System.out.print("\n" + i);
			for(int j = (i % (60000/batchSize)) * batchSize; j < ((i % (60000/batchSize)) + 1) * batchSize; j++) {
				//forward
				output3d = getImg(mnistMatrix, j, convLayers[0].input.length);
				for(int k = 0; k < convLayers.length; k++) {
					output3d = convLayers[k].forward(output3d);
					output3d = poolLayers[k].forward(output3d);
					output3d = activLayers[k].forward(output3d);
				}
				output = reshape.forward(output3d);
				for(int k = 0; k < denseLayers.length; k++) {
					output = denseLayers[k].forward(output);
					output = activLayers[k + convLayers.length].forward(output);
				}
				error = msePrime(output, mnistMatrix[j].getLabel());
				//backward
				for(int k = denseLayers.length - 1; k >= 0; k--) {
					error = activLayers[k + convLayers.length].backward(error);
					error = denseLayers[k].backward(error);
				}
					error3d = reshape.backward(error);
				for(int k = convLayers.length - 1; k >=0; k--) {
					error3d = activLayers[k].backward(error3d);
					error3d = poolLayers[k].backward(error3d);
					error3d = convLayers[k].backward(error3d);
				}
			}
			for(int j = 0; j < denseLayers.length; j++) {
				denseLayers[j].updateConfig(learningRate);
			}
			for(int j = 0; j < convLayers.length; j++) {
				convLayers[j].updateConfig(learningRate);
			}
		}

		//==============================================================================
		
		//sprawdzenie sieci
		int answer;
		int numOfGoodAnswers = 0;
		int tests = 0;
		System.out.print("\n\nTesting network:");
		for(int i = 0; i < test.length; i++) {
			//forward
			output3d = getImg(test, i, convLayers[0].input.length);
			for(int k = 0; k < convLayers.length; k++) {
				output3d = convLayers[k].forward(output3d);
				output3d = poolLayers[k].forward(output3d);
				output3d = activLayers[k].forward(output3d);
			}
			output = reshape.forward(output3d);
			for(int k = 0; k < denseLayers.length; k++) {
				output = denseLayers[k].forward(output);
				output = activLayers[k + convLayers.length].forward(output);
			}
			System.out.print("\nNumber "+i+"\nLabel: " + test[i].getLabel() + "\n");
			answer = printAnswer(output);
			if(answer == test[i].getLabel()) numOfGoodAnswers++;
			printVector(output);
			if(answer != test[i].getLabel()) drawImg(getImg(test, i, convLayers[0].input.length));
			tests++;
		}
		System.out.print("\n\nNetwork testing finished.\nNumber of tests:        " 
		 +tests + "\nNumber of good answers: " + numOfGoodAnswers);
		
	}//end of main() method
	//mse
	static double[] msePrime(double[] results, double[] correct) {
		double[] out = new double[results.length];		
		for(int i = 0; i < out.length; i++) {
			out[i] = (2*(results[i] - correct[i]));//test wyjebania /out.length;
		}
		return out;
	}//end of msePrime() method
	static double[] msePrime(double[] results, int correct) {
		double[] corr = new double[results.length];
		for(int i = 0; i < corr.length; i++) {
			if(i == correct) corr[i] = 1;
			else corr[i] = 0;
		}
		double[] out = new double[results.length];		
		for(int i = 0; i < out.length; i++) {
			out[i] = (2*(results[i] - corr[i]));//test wyjebania /out.length;
		}
		return out;
	}//end of msePrime() method
	//obrazki
	static void drawImg(double[][][] img) {
		char[] signs = {' ', '.', ':', ';', '-', '=', '+', '*', '#', '%', '@'};
		for(int i = 0; i < img.length; i++) {
			for(int j = 0; j < img[i].length; j++) {
				for(int k = 0; k < img[i][j].length; k++) {
					if((int)(img[i][j][k]*10) >= 10)System.out.print('@');
					else if((int)(img[i][j][k]*10) <= 0)System.out.print(' ');
					else System.out.print(signs[(int)(img[i][j][k]*10)]);
				}
			System.out.print("\n");
			}
		System.out.print("\n");
		}
	}//end of drawImg() method
	static void drawImg(double[][] img) {
		char[] signs = {' ', '.', ':', ';', '-', '=', '+', '*', '#', '%', '@'};
		for(int i = 0; i < img.length; i++) {
			System.out.print("\n");
			for(int j = 0; j < img[i].length; j++) {
					if((int)(img[i][j]*10) >= 10)System.out.print('@');
					else if((int)(img[i][j]*10) <= 0)System.out.print(' ');
					else System.out.print(signs[(int)(img[i][j]*10)]);
			}
		}
	}//end of drawImg() method
	static int printAnswer(double[] results) {
		double maxValue = 0;
		int maxIndex = 0;
		for(int i = 0; i < results.length; i++) {
			if(results[i] > maxValue) {
				maxIndex = i;
				maxValue = results[i];
				}
		}
		System.out.print("Network answer: " + maxIndex);
		return maxIndex;
	}//end of printAnswer() method
	static void printConfig(DenseLayer layer) {
		System.out.print("\nBias: ");
		printVector(layer.bias);
		System.out.print("\nWeights: ");
		for(int i = 0; i < layer.weights.length;i++) printVector(layer.weights[i]);
	}//end of printConfig() method
	static void printConfig(ConvLayer layer, boolean picture) {
		System.out.print("\nBias: ");
		for(int i = 0; i < layer.bias.length; i++){
			System.out.print("\nMatrix " + i);
			if(picture == true) drawImg(layer.bias[i]);
			else printMatrix(layer.bias[i]);
			}
		System.out.print("\nKernels: ");
		for(int i = 0; i < layer.kernel.length; i++){
			System.out.print("\nKernel " + i);
			for(int j = 0; j <layer.kernel[i].length; j++) {
				System.out.print("\nMatrix " + j);
				if(picture == true) drawImg(layer.bias[i]);
				else printMatrix(layer.kernel[i][j]);
			}
		}
	}//end of printConfig() method
	static double[] getImg(MnistMatrix[] mnistMatrix, int num){
		double[] img = new double[mnistMatrix[num].getNumberOfRows() * mnistMatrix[num].getNumberOfColumns()];
		int i = 0;
		double value = 0;
		for(int r = 0; r < mnistMatrix[num].getNumberOfRows(); r++) {
			for(int c = 0; c < mnistMatrix[num].getNumberOfColumns(); c++) {
				value = mnistMatrix[num].getValue(r, c);
				img[i] = value/255;
				i++;
			}
		}
		return img;
	}//end of getImg() method
	static double[][][] getImg(MnistMatrix[] mnistMatrix, int num, int depth){
		double[][][] img = new double[depth][mnistMatrix[num].getNumberOfRows()][mnistMatrix[num].getNumberOfColumns()];//new double[mnistMatrix[num].getNumberOfRows() * mnistMatrix[num].getNumberOfColumns()];
		double value = 0;
		for(int i = 0; i < depth; i++) {
			for(int r = 0; r < mnistMatrix[num].getNumberOfRows(); r++) {
				for(int c = 0; c < mnistMatrix[num].getNumberOfColumns(); c++) {
					value = mnistMatrix[num].getValue(r, c);
					img[i][r][c] = value/255;
				}
			}
		}
		return img;
	}//end of getImg() method
	static void printVector(double[] output) {
		System.out.print("\n");
		for(int i = 0; i < output.length; i++) {
			System.out.printf("%f ", output[i]);
			//System.out.print("\n" + i + ": " + output[i]);
		}
	}//end of printVector() method
	static void printMatrix(double[][] matrix) {
		System.out.print("\n");
		for(int i = 0; i < matrix.length; i++) {
			System.out.print("\n");
			for(int j = 0; j < matrix[i].length; j++) {
				System.out.printf("%f ", matrix[i][j]);
			}
		}
	}//end of printMatrix() method
	static void printMatrix3d(double[][][] matrix) {
		System.out.print("\n");
		for(int i = 0; i < matrix.length; i++) {
			System.out.print("\n\n");
			for(int j = 0; j < matrix[i].length; j++) {
				System.out.print("\n");
				for(int k = 0; k < matrix[i][j].length; k++) {
					System.out.printf("%f ", matrix[i][j][k]);
				}
			}
		}
	}//end of printMatrix() method
	static void drawMatrix3d(double[][][] matrix) {
		char[] signs = {' ', '.', ':', ';', '-', '=', '+', '*', '#', '%', '@'};
		System.out.print("\n");
		for(int i = 0; i < matrix.length; i++) {
			System.out.print("\n\n");
			for(int j = 0; j < matrix[i].length; j++) {
				System.out.print("\n");
				for(int k = 0; k < matrix[i][j].length; k++) {
					System.out.print(signs[(int)(matrix[i][j][k]*10)]);
				}
			}
		}
	}//end of printMatrix() method
}//end of class
