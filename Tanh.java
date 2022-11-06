
public class Tanh extends ActivLayer{
	Tanh(int inputLen){
		super(inputLen);
	}
	Tanh(int depth, int inputLen){
		super(depth, inputLen);
	}
	
	double activate(double x) {
		return Math.tanh(x);
	}//end of activate() method
	double activateDerivative(double x) {
		return 1 - Math.pow(Math.tanh(x), 2);
	}//end of activateDerivative() method
}//end of class
