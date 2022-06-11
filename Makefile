hist: hist.cpp hist.hpp hist.cl
	@$(CXX) -Wall -std=c++11 hist.cpp -o hist -lOpenCL
	@./hist

clean:
	rm hist