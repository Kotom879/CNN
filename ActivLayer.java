
public class ActivLayer {
	double[] input;
	double[] output;
	
	double[][][] input3d;
	double[][][] output3d;
	
	ActivLayer(int inputLen){
		this.input = new double[inputLen];
		this.output = new double[inputLen];
	}
	ActivLayer(int depth, int inputLen){
		this.input3d = new double[depth][inputLen][inputLen];
		this.output3d = new double[depth][inputLen][inputLen];
	}
	double[] forward(double[] input) {
		this.input = input;
		for(int i = 0; i < output.length; i++) {
			this.output[i] = activate(input[i]);
		}
		return output;
	}//end of forward(double[]) method
	double[][][] forward(double[][][] input){
		this.input3d = input;
		for(int i = 0; i < input.length; i++) {
			for(int j = 0; j < input[i].length; j++) {
				for(int k = 0; k < input[i][j].length; k++) {
					output3d[i][j][k] = activate(input3d[i][j][k]);
				}
			}
		}
		return output3d;
	}//end of forward(double[][][]) method
	double[] backward(double[] error) {
		double[] inputGradient = new double[input.length];
		for(int i = 0; i < inputGradient.length; i++) {
			inputGradient[i] = error[i] * activateDerivative(input[i]);
		}
		return inputGradient;
	}//end of backward(double[]) method
	double[][][] backward(double[][][] error){
		double[][][] inputGradient = new double[input3d.length][input3d[0].length][input3d[0][0].length];
		for(int i = 0; i < error.length; i++) {
			for(int j = 0; j < error[i].length; j++) {
				for(int k = 0; k < error[i][j].length; k++) {
					inputGradient[i][j][k] = error[i][j][k] * activateDerivative(input3d[i][j][k]);
				}
			}
		}
		return inputGradient;
	}//end of backward(double[][][]) method
	double activate(double x) {
		return 1 / (1 + Math.exp(-x));
	}//end of sigmoid() method
	double activateDerivative(double x) {
		return activate(x) * (1 - activate(x));
	}//end of sigmoidDerivative() method
}//end of class
