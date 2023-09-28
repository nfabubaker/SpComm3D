testSDDMM: 
	g++ -ggdb -o testSDDMM miniapp/main_random_mat.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
sddmm:
	mpic++ --std=c++17 -O3 -o sddmm miniapp/main.cpp src/*.cpp
	#g++ -O3 -o sddmm miniapp/main.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
debug: 
	mpic++ --std=c++17 -ggdb -o sddmm miniapp/main.cpp src/*.cpp
	#g++ -O3 -o sddmm miniapp/main.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
serialSim: 
	g++ -ggdb -o serialSim miniapp/serialSim.cpp
db:
	g++ -ggdb -o db tools/read_distribute_mm.cpp
clean: 
	rm sddmm serialSim debug db
