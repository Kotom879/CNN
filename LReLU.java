
public class LReLU extends ActivLayer{
	LReLU(int inputLen){
		super(inputLen);
	}
	LReLU(int depth, int inputLen){
		super(depth, inputLen);
	}

	double activate(double x) {
		if(x >= 0) return x;
		else return 0.01 * x;
	}//end of activate() method
	double activateDerivative(double x) {
		if(x >= 0)return 1;
		else return -0.01;
	}//end of activateDerivative() method
}//end of class