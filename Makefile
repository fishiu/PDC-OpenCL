sharp: sharp.cpp sharp.hpp sharp.cl
	@$(CXX) -Wall -g -std=c++14 sharp.cpp -o sharp.bin -lOpenCL
	@./sharp.bin

hist: hist.cpp hist.hpp hist.cl
	@$(CXX) -Wall -g -std=c++14 hist.cpp -o hist.bin -lOpenCL
	@./hist.bin

clean:
	rm sharp.bin hist.bin