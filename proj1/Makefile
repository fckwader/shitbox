CXXFLAGS=-O2

OBJS = main.o

all: multigrid

multigrid: $(OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OBJS) -o multigrid

clean:
	rm -f *~ *.o *.vtk multigrid

