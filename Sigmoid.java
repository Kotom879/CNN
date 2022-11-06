
public class Sigmoid extends ActivLayer{
	Sigmoid(int inputLen){
		super(inputLen);
	}
	Sigmoid(int depth, int inputLen){
		super(depth, inputLen);
	}

	double activate(double x) {
		return 1 / (1 + Math.exp(-x));
	}//end of activate() method
	double activateDerivative(double x) {
		return activate(x) * (1 - activate(x));
	}//end of activateDerivative() method
}//end of class
