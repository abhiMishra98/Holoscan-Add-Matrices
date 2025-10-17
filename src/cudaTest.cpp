#include <holoscan/holoscan.hpp>
#include "cudaKernel.h"
#include <cuda_runtime.h>
#include <iostream>

namespace holoscan::ops{
    class opInit: public Operator{
        private:
            int *a,*b;
            int N;
        public:
            HOLOSCAN_OPERATOR_FORWARD_ARGS(opInit);
            // To check if this way of declaring a constructor is fine or not
            opInit(){
                N= 1<<20;
            }

            void setup(OperatorSpec &spec) override{
                spec.output<int *>("outFirst");
                spec.output<int *>("outSecond");
                spec.output<int>("outNumbers");
            }

            void compute(InputContext &in, OutputContext &op, ExecutionContext &ex) override{
                //To check if CUDA APIs can be called or not and also if initMatrix can be called successfully
                cudaMallocManaged(&a,sizeof(int)*N);
                cudaMallocManaged(&b,sizeof(int)*N);
                initMatrix(a,b,N);

                op.emit(a,"outFirst");
                op.emit(b,"outSecond");
                op.emit(N,"outNumbers");
            }
    };

    class opOut: public Operator{
        public:
            HOLOSCAN_OPERATOR_FORWARD_ARGS(opOut);
            opOut() = default;

            void setup(OperatorSpec &spec) override{
                spec.input<int *>("inFirst");
                spec.input<int *>("inSecond");
                spec.input<int>("inNumbers");
            }

            void compute(InputContext &ip, OutputContext &op, ExecutionContext &ex) override{
                auto ipFirst = ip.receive<int *>("inFirst");
                auto ipSecond = ip.receive<int *>("inSecond");
                auto inNumbers = ip.receive<int>("inNumbers");

                int *a = ipFirst.value();
                int *b = ipSecond.value();
                int N = inNumbers.value();
                addMatrix(a,b,N);
            }
            
    };
}

class perfCal: public holoscan::Application{
    public:
        void compose() override{
            using namespace holoscan;
            auto opInitOperator = make_operator<ops::opInit>("opInit",make_condition<CountCondition>(1));
            auto opOutOperator = make_operator<ops::opOut>("opOut",make_condition<CountCondition>(1));

            add_operator(opInitOperator);
            add_operator(opOutOperator);

            std::set<std::pair<std::string,std::string>> port_pairs = {
                {"outFirst","inFirst"},{"outSecond","inSecond"},{"outNumbers","inNumbers"}
            };

            add_flow(opInitOperator,opOutOperator,port_pairs);
        }
};

int main() {
    auto myApp = holoscan::make_application<perfCal>();
    myApp->run();
    return 0;
}