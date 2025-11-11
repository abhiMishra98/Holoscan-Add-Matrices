#include <holoscan/holoscan.hpp>
#include "cudaKernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
         if(err != cudaSuccess){ \
             std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                       << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
             exit(EXIT_FAILURE); \
         } \
    } while(0)


namespace holoscan::ops{
    class opInit: public Operator{
        private:
            int *a = nullptr;
            int *b = nullptr;
            int N;
            std::shared_ptr<int> num_ptr;
            
        public:
            HOLOSCAN_OPERATOR_FORWARD_ARGS(opInit);
            // To check if this way of declaring a constructor is fine or not
            void start() override {
                
                    N= 1<<20;
                // num_ptr = std::make_shared<int>(N);

            }


            void setup(OperatorSpec &spec) override{
                spec.output<int *>("outFirst");
                spec.output<int *>("outSecond");
                spec.output<int>("outNumbers");
            }

            void compute(InputContext &in, OutputContext &op, ExecutionContext &ex) override{
                //To check if CUDA APIs can be called or not and also if initMatrix can be called successfully
                cudaSetDevice(0);
                a = new int[N];
                b = new int[N];
                initMatrix(a,b,N);

               // cudaMallocManaged(&a,sizeof(int)*N);
               // cudaMallocManaged(&b,sizeof(int)*N);
                
                op.emit(a,"outFirst");
                op.emit(b,"outSecond");
                //op.emit(num_ptr,"outNumbers");
                std::cout<<"N is "<<N<<std::endl;
                op.emit(N, "outNumbers");
            }

            void stop() override {
                if(a){
                    delete[] a;
                }
                if(b){
                    delete[] b;
                }
                
            }
    };

    class opOut: public Operator{
        private:
            int *d_a,*d_b;
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

               int No = (inNumbers.value());
                std::cout<<"N is "<<No<<std::endl;
                CUDA_CHECK(cudaMalloc(&d_a, No * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_b, No * sizeof(int)));
                int *a = ipFirst.value();
                int *b = ipSecond.value();
                CUDA_CHECK(cudaMemcpy(d_a,a,No*sizeof(int),cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_b,b,No*sizeof(int),cudaMemcpyHostToDevice));

                addMatrix(d_a,d_b,No);

                cudaDeviceSynchronize();
                cudaFree(d_a);
                cudaFree(d_b);
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