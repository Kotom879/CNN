import java.util.Random;

public class DenseLayer {
	double[] input;
	double[] output;
	
	double[][] weights;
	double[] bias;
	
	double[][] weightsGradient;
	double[] biasGradient;
	
	DenseLayer(int inputLen, int outputLen){
		this.input = new double[inputLen];
		this.output = new double[outputLen];
		this.weights = new double[outputLen][inputLen];
		this.bias = new double[outputLen];
		this.biasGradient = new double[outputLen];
		this.weightsGradient = new double[outputLen][inputLen];
	}
	void randomize() {
		Random rand = new Random();
		for(int i = 0; i < output.length; i++) {
			bias[i] = rand.nextDouble()  - 0.5;
			for(int j = 0; j < input.length; j++) {
				weights[i][j] = rand.nextDouble() - 0.5;
				if(weights[i][j] == 0) weights[i][j] += 0.1;
			}
		}
	}//end of randomize() method
	double[] forward(double[] input) {
		this.input = input;
		this.output = new double[output.length];
		for(int i = 0; i < output.length; i++) {
			for(int j = 0; j < input.length; j++) {
				output[i] += input[j] * weights[i][j]; 
			}
			output[i] = output[i] + bias[i];
		}
		return output;
	}//end of forward() method
	double[] backward(double[] error) {
		for(int i = 0; i < this.weightsGradient.length; i++) {
			this.biasGradient[i] += error[i];
			for(int j = 0; j < this.weightsGradient[i].length; j++) {		
				this.weightsGradient[i][j] += error[i] * input[j];
			}
		}
		return getInputGrad(error);//inputGradient;
	}//end of backward() method
	void updateConfig(double learningRate) {
		for(int i = 0; i < this.output.length; i++) {
			this.bias[i] -= this.biasGradient[i] * learningRate;
			for(int j = 0; j < this.input.length; j++) {
				this.weights[i][j] -= this.weightsGradient[i][j] * learningRate;
			}
		}
		this.biasGradient = new double[output.length];
		this.weightsGradient = new double[output.length][input.length];
	}//end of updateConfig() method
	double[] getInputGrad(double[] error) {
		double[] inputGradient = new double[input.length];
		for(int i = 0; i < inputGradient.length; i++) {
			for(int j = 0; j < output.length; j++) {
				inputGradient[i] += error[j] * weights[j][i]; 
			}
		}
		return inputGradient;
	}//end of getInputGrad() method
}//end of DenseLayer() class
