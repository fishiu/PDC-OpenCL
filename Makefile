sharp: sharp.cpp sharp.hpp sharp.cl
	@$(CXX) -Wall -std=c++11 sharp.cpp -o sharp -lOpenCL
	@./sharp

hist: hist.cpp hist.hpp hist.cl
	@$(CXX) -Wall -std=c++11 hist.cpp -o hist -lOpenCL
	@./hist

clean:
	rm sharp hist