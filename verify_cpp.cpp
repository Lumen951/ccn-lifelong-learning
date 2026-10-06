// Deterministic probes use the original C++ implementation, without a database.
#include <iomanip>
#include <iostream>
#include "include/nn/lstm.h"
#include "include/nn/networks/lstm_bptt.h"

template<class T> void emit(const T& values) {
  for (auto v : values) std::cout << v << ' ';
}

class CellProbe : public LSTM {
 public:
  CellProbe() : LSTM(.02f,.03f,-.01f,.04f,.01f,-.02f,.03f,.02f,.1f,.999f) {
    reset_state();
  }
  void dump() {
    std::cout << "cell " << h << ' ' << c << ' ';
    emit(w_i); emit(w_f); emit(w_g); emit(w_o);
    emit(Gw_i); emit(Gw_f); emit(Gw_g); emit(Gw_o);
    std::cout << Gu_i << ' ' << Gu_f << ' ' << Gu_g << ' ' << Gu_o << ' '
              << Gb_i << ' ' << Gb_f << ' ' << Gb_g << ' ' << Gb_o << ' '
              << u_i << ' ' << u_f << ' ' << u_g << ' ' << u_o << ' '
              << b_i << ' ' << b_f << ' ' << b_g << ' ' << b_o << '\n';
  }
};

int main() {
  std::cout << std::setprecision(10);
  LinearNeuron x0(true,false), x1(true,false), x2(true,false);
  CellProbe cell;
  Neuron* inputs[] = {&x0, &x1, &x2};
  for (int j=0;j<3;j++)
    cell.add_synapse(inputs[j], .01f*(j+1), -.02f*(j+1), .03f*(j+1), -.01f*(j+1));
  for (int t=0;t<8;t++) {
    for (int j=0;j<3;j++) inputs[j]->value = (t+j)%2;
    cell.update_value_sync(); cell.fire();
    cell.decay_gradient(.891f);
    cell.compute_gradient_of_all_synapses(); cell.accumulate_gradient(.3f);
    cell.update_weights(.001f, -.2f);
    cell.dump();
  }
  DenseLSTMRmsProp net(.001f,0,2,3,8,.9999f,1e-8f);
  for (int j=0;j<net.W.size();j++) net.W[j]=.01f*((j%7)-3);
  for (int j=0;j<net.U.size();j++) net.U[j]=.01f*((j%5)-2);
  for (int j=0;j<net.b.size();j++) net.b[j]=.01f*((j%3)-1);
  net.prediction_weights = {.2f,-.1f};
  // No buffer wrap here: Python intentionally fixes the original boundary bug.
  for (int t=0;t<3;t++) {
    float prediction = net.forward({float(t%2),float((t+1)%2),float(t%2)});
    net.decay_gradient(.891f); net.backward(0); net.update_parameters(0,-.2f);
    std::cout << "dense " << prediction << ' '; emit(net.get_state());
    emit(net.W); emit(net.U); emit(net.b); emit(net.prediction_weights);
    emit(net.W_grad); emit(net.U_grad); emit(net.b_grad); emit(net.prediction_weights_grad);
    std::cout << '\n';
  }
}
