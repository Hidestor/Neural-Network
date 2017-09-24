/* @ author Shikhar Sharma
     15530 NIT Hamirpur

Implementation for Feedforward + Backpropagation network (Delta Learning Rule)
Applicable only for Single layer neural network having single o/p
*/

#include <iostream>
#include <math.h>
#define e 2.7182

using namespace std;

class Neuron
{
 public: 
  double *w_; // weight of inputs
  double *b_; // bias

  double *input_, output_; //saved for back propagation

 public: 
  
  Neuron(int n){  //constructor
    w_ = new double(n);
    b_ = new double(n);
    input_ = new double(n);
    //output_ = new double(n);
    for(int i=0;i<n;i++){
      cout<<"Weight for Neuron "<<i+1<<": ";
      cin>>w_[i];
      cout<<"Bias for Neuron "<<i+1<<": ";
      cin>>b_[i];
      cout<<endl;
    }  
  }

  double get_activation(double x, char ch)
  {
     //for Binary Sigmoidal Function f(x) = 1 / (1+e^-x);
     if(ch=='1'){
      double denominator = 1.0 + (pow(e,-x));   
      return 1/denominator; 
     }

     //for Bipolar Sigmoidal Function f(x) = (1-e^-x) / (1+e^-x);
     else{
      double denominator = 1.0 + (pow(e,-x));
      double numerator = 1.0 - (pow(e,-x));
      return numerator/denominator;
     }
  }

  double feed_forward(double* _input, int n, char ch)
  {
    for(int i=0;i<n;i++){
      input_[i] = _input[i];  
    }
    
    // for multiple inputs,
    // \sigma = w0_ * x0_ + w1_ * x0_ ... + b
    
    double sigma = b_[0];    
    
    for(int i=0;i<n;i++){
      sigma += w_[i]*_input[i]; 
    } 

    output_ = get_activation(sigma,ch);
    return output_; 
  }

  double get_activation_gradient(double x, char ch)
  {
    if(ch==1){
      return (get_activation(x,ch) * (1 - get_activation(x,ch)));
    }
    else{
      return 0.5*((1 - get_activation(x,ch))*(1 + get_activation(x,ch)));
    }
  }

  void propagate_backward(double target, int n, char ch)
  {
    const double alpha = 0.1; //learning rate
    const double grad = (target - output_ ) * get_activation_gradient(output_,ch);

    for(int i=0;i<n;i++){
      w_[i] += alpha * grad * input_[i]; // last input_ came from d
      b_[i] += alpha * grad * 1.0; // last 1.0 came from d(wx+b)/db =1  
    }  
  }

  void feed_forward_and_print(double *input,int n, char ch)
  {
    cout<<"[";
    for(int i=0;i<n;i++){
      cout << input[i] <<" ";
    }
    cout<<"] --> ";
    
    cout<<feed_forward(input,n,ch)<<"\n";
  }

};

int main()
{
  int n, target;
  cout<<"Enter the no. of Neurons in Network: ";
  cin>>n;
  cout<<endl;
  Neuron my_neuron(n);

  double *input = new double(n);  //input for each neurons

  for(int i=0;i<n;i++){
    cout<<"Input for Neuron "<<i+1<<": ";
    cin>>input[i];
  }

cout<<"\nTarget Value: ";
cin>>target;
cout<<"\n";

  char ch,more='y';
  cout<<"\nType of Activation Function: \n";
  cout<<"1) Binary Sigmoidal Function\n";
  cout<<"2) Bipolar Sigmoidal Function\n";
  cout<<"Enter your choice: ";
  cin>>ch;

  for (int r=0; r < 10; r++)
  {
    cout << "\n\nTraining " << r <<endl;
    my_neuron.feed_forward_and_print(input,n,ch);
    my_neuron.propagate_backward(target, n, ch);
    cout<<"w = ";
    for(int i=0;i<n;i++){
      cout<<my_neuron.w_[i] << " ";  
    }
    
    cout<<"\nb = ";

    for(int i=0;i<n;i++){
      cout<<my_neuron.b_[i] << " ";  
    }
  }
  cout<<endl;
  return 0;
}
