FLAGS = -Wall -Wextra  -pedantic


all: integrate

monte.o: monte.cpp
	g++ -c -cuda -O4 $(FLAGS) monte.cpp -o monte.o

integrate: monte.o
	g++ monte.o -o integrate

run: integrate
	./integrate 0 1 100000000 6

run_many: integrate
	for num in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36; do \
		/home/kchiu/time -p ./integrate 0 1 100000000 $$num ; \
	done

srun: integrate
	srun -N 1 ./integrate -1 3 100000000 6

clean:
	rm *.o integrate

valgrind: integrate
	valgrind ./integrate 0 1 1000000 8
