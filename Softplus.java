
public class Softplus extends ActivLayer {
		Softplus(int inputLen){
			super(inputLen);
		}
		Softplus(int depth, int inputLen){
			super(depth, inputLen);
		}

		double activate(double x) {
			return Math.log(1 + Math.exp(x));
		}//end of activate() method
		double activateDerivative(double x) {
			return 1 / (1 + Math.exp(-x));
		}//end of activateDerivative() method
}//end of class
