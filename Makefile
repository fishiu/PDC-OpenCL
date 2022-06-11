sharp: sharp.cpp sharp.hpp sharp.cl
	@$(CXX) -Wall -std=c++11 sharp.cpp -o sharp.bin -lOpenCL
	@./sharp.bin

hist: hist.cpp hist.hpp hist.cl
	@$(CXX) -Wall -std=c++11 hist.cpp -o hist.bin -lOpenCL
	@./hist.bin

clean:
	rm sharp.bin hist.bin