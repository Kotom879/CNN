
public class Reshape {
	double[][][] input;
	double[] output;
	
	Reshape(int depth, int inputSize){
		this.input = new double[depth][inputSize][inputSize];
		this.output = new double[depth*inputSize*inputSize];
	}
	double[] forward(double[][][] input) {
		this.input = input;
		this.output = new double[input.length * input[0].length * input[0][0].length];
		int x = 0;
		for(int i = 0; i < input.length; i++) {
			for(int j = 0; j < input[i].length; j++) {
				for(int k = 0; k < input[i][j].length; k++) {
					output[x] = input[i][j][k];
					x++;
				}
			}
		}
		return output;
	}//end of forward() method
	double[][][] backward(double[] error){
		double[][][] out = new double[input.length][input[0].length][input[0][0].length];
		int x = 0;
		for(int i = 0; i < input.length; i++) {
			for(int j = 0; j < input[i].length; j++) {
				for(int k = 0; k < input[i][j].length; k++) {
					out[i][j][k] = error[x];
					x++;
				}
			}
		}
		return out;
	}//end of backward() method
}//end of reshape class
