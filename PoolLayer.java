
public class PoolLayer {
	double[][][] input;
	double[][][] inputMap;
	double[][][] output;
	
	PoolLayer(int depth, int inputSize){
		this.input = new double[depth][inputSize][inputSize];
		this.inputMap = new double[depth][inputSize][inputSize];
		this.output = new double[depth][inputSize / 2][inputSize / 2];
	}
	double[][][] forward(double[][][] input){
		this.input = input;
		int outputSize = input[0].length / 2;
		this.inputMap = new double[input.length][input[0].length][input[0][0].length];
		this.output = new double[input.length][outputSize][outputSize];
		for(int i = 0; i < output.length; i++) {
			for(int j = 0; j < output[i].length; j++) {
				for(int k = 0; k < output[i][j].length; k++) {
					double max = Double.NEGATIVE_INFINITY;
					int mapx = 0;
					int mapy = 0;
					for(int x = 0; x < 2; x++) {
						for(int y = 0; y < 2; y++) {
							if(input[i][j * 2 + x][k * 2 + y] > max) {
								max = input[i][j * 2 + x][k * 2 + y]; 
								mapx = j * 2 + x;
								mapy = k * 2 + y;		
								}
							}
						}
					inputMap[i][mapx][mapy] = 1;
					output[i][j][k] = max;
					}
				}
			}
		return this.output;
	}//end of forward() method
	double[][][] backward(double[][][] error){
		double[][][] inputGradient = new double[input.length][input[0].length][input[0][0].length];
		int n = 0;
		for(int i = 0; i < output.length; i++) {
			for(int j = 0; j < output[i].length; j++) {
				for(int k = 0; k < output[i][j].length; k++) {
					for(int x = 0; x < 2; x++) {
						for(int y = 0; y < 2; y++) {
							if(inputMap[i][j * 2 + x][k * 2 + y] == 1) {
								inputGradient[i][j * 2 + x][k * 2 + y] = getElement(error, n);
								n++;
							}
						}
					}
				}
			}
		}
		return inputGradient;
	}//end of backward() method
	double getElement(double[][][] matrix, int n){
		int x = 0;
		for(int i = 0; i < matrix.length; i++) {
			for(int j = 0; j < matrix[i].length; j++) {
				for(int k = 0; k < matrix[i][j].length; k++) {
					if(x == n) return matrix[i][j][k];
					x++;
				}
			}
		}
		return 0;
	}//end of getElement() method
}//end of class
