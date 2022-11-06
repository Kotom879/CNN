
public class ReLU extends ActivLayer{
	ReLU(int inputLen){
		super(inputLen);
	}
	ReLU(int depth, int inputLen){
		super(depth, inputLen);
	}
	
	double activate(double x) {
		if(x >= 0) return x;
		else return 0;
	}//end of activate() method
	double activateDerivative(double x) {
		if(x >= 0)return 1;
		else return 0;
	}//end of activateDerivative() method
}//end of class
