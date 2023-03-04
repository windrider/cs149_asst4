## cs149_asst4

​	使用OpenMP加速**pagerank**算法和各种**BFS**算法

## Part 1: Warm up: Implementing Page Rank

​	实现Page Rank算法并使用OpenMP加速。

​	Page Rank算法会不停迭代计算网络图中每个节点（网站）的权重，并将每轮迭代前后的权重相减,若计算出的 global_diff 小于某个阈值则算法收敛：

```c++
//global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
//converged = (global_diff < convergence)
```

​	根据文档中对算法的描述实现page rannk算法，用OpenMP并行计算每个点的新权重，并使用#pragma omp parallel for reduction(+:global_diff)将计算出的权重差累积到共享变量 global_diff 中。

```c++
// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double *score_new=new double[numNodes];
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }

  bool converged=false;
  Vertex* nullout=new Vertex[numNodes];
  int nulloutsize=0;
  for(Vertex v=0;v<numNodes;++v){
    if(outgoing_size(g,v)==0){
      nullout[nulloutsize]=v;
      ++nulloutsize;
    }
  }
  while(!converged){
    double global_diff=0.0;
    #pragma omp parallel for reduction(+:global_diff)
    for(Vertex vi=0;vi<numNodes;++vi){
      score_new[vi]=0.0;
      const Vertex* start = incoming_begin(g, vi);
      const Vertex* end = incoming_end(g, vi);
      for (const Vertex* v=start; v!=end; v++){
        score_new[vi]+=solution[*v]/outgoing_size(g,*v);
      }
      score_new[vi]=(damping*score_new[vi])+(1.0-damping)/numNodes;
      for( int i=0;i<nulloutsize;i++){
        score_new[vi]+= damping*solution[nullout[i]]/numNodes;
      }
      global_diff+=abs(score_new[vi]-solution[vi]);
      
    }
    converged=(global_diff<convergence);
    memcpy(solution,score_new,numNodes*sizeof(double));
  }
  free(nullout);
  free(score_new);
}
```

使用 omp_set_num_threads(thread_count) 设置线程数量，最后在不同线程下page_rank算法的运行时间如下：

![2808450a49ebc48e753651f49fc41c7](graphs/2808450a49ebc48e753651f49fc41c7.png)

可以看到运行速度基本上是与线程数成正比的。

## Part 2: Parallel "Top Down" Breadth-First Search 

​	题目中顺序计算自顶向下的BFS算法的串行计算代码已经给出，要求用openMP加速该算法。

​	操作比较简单，在最外层循环加上# pragma omp parallel for即可，但有一点需要注意，

​	int index=new_frontier->count++;这一句如果被所有线程不加限制地异步计算会导致count计数发生错误。所以需要加上#pragma omp critical限制同时只能有一条指令执行该语句。

​	最后修改的代码如下（frontier为入参，new_frontier为出参，每轮从frontier找邻接节点放入new_frontier）：

```c++
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    #pragma omp parallel for
    for (int i=0; i<frontier->count; i++) {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];
        int mynum=end_edge-start_edge;
        
        // attempt to add all neighbors to the new frontier
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                if (distances[outgoing] == NOT_VISITED_MARKER) {
                    distances[outgoing] = distances[node] + 1;
                    int index;
                    #pragma omp critical
                    index=new_frontier->count++;
                    new_frontier->vertices[index] = outgoing;
                    
                }
            }
        //}
    }
}
```

​	虽然题目文档中说了**#pragma omp critical**和**#pragma omp atomic**都可以起到控制线程对共享变量修改的作用，但实际操作中发现使用atomic只能作用于形如“x++”的语句，不能作用于赋值语句，但index涉及到对后续共享变量的修改，同样需要放在临界区中，所以用critical更能保证结果正确。

​	最后实验的结果如下：

![f1dbdb10ad4d52ab8e3376e5f950a1d](graphs/f1dbdb10ad4d52ab8e3376e5f950a1d.png)

![9bb1d132f4907fc79dceb247efde6de](graphs/9bb1d132f4907fc79dceb247efde6de.png)

可以看到从单线程到双线程性能提高了，但是到4个线程时并没有让程序更快。这是因为**#pragma omp critical**不可避免地会造成线程阻塞，线程数量增加，阻塞的次数也会增加。



尝试使用CAS实现无锁：

```c++
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    #pragma omp parallel for
    for (int i=0; i<frontier->count; i++) {
        int node = frontier->vertices[i];
        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                if (distances[outgoing] == NOT_VISITED_MARKER) {
                    distances[outgoing] = distances[node] + 1;
                    while(!__sync_bool_compare_and_swap(new_frontier->vertices+new_frontier->count,NEWFRONTIER_OLD,outgoing));
                    #pragma omp atomic
                    new_frontier->count++;
                }
            }
    }
}
```

程序运行效率有了一些微小的提高：

![1454c592e9075d6c41d299479735bb0](graphs/1454c592e9075d6c41d299479735bb0.png)

 

 后续更新：CAS那里直接写成

```c++
while(!__sync_bool_compare_and_swap(new_frontier->vertices+new_frontier->count++,NEWFRONTIER_OLD,outgoing));
```

可以提升到这个程度

​	![56bd5f7dd3e732522df7cdb70a14fe8](graphs/56bd5f7dd3e732522df7cdb70a14fe8.png)



另外，该函数有两层循环，实验中尝试过只将**pragma omp parallel**加到内层循环上，发现程序执行得很慢，并且线程越多会越慢

![648eda5628e3a6db52222dbd10bf9be](graphs/648eda5628e3a6db52222dbd10bf9be.png)

​	这是因为内层循环体只有很简短的计算和赋值，导致线程切换得很快，线程切换也会有开销，并且会有更频繁地阻塞。而且这样的设计会让线程之间频繁访问不连续的内存空间，弄脏cache。

## Part 3: "Bottom Up" BFS

要求实验者自行实现自底向上的BFS算法，并对其进行加速优化。

在题目的文档中给出了算法的思路：

```
for each vertex v in graph:
    if v has not been visited AND 
       v shares an incoming edge with a vertex u on the frontier:
          add vertex v to frontier;
```

​		一开始的想法是维护一个unvisited_set存储所有还没有访问过的节点，当节点一旦被访问过就从unvisited_set中去除（将这轮访问到的节点放入new_frontier,将没访问过的节点放入new_unvisited_set，最后交换unvisited_set和new_unvisited_set）。并且模仿上个part的操作，维护一个从frontier到new_frontier的过程。

```c++
void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    vertex_set* unvisited_set,
    vertex_set* new_unvisited_set,
    int* distances)
{
    int mydistance=0;
    #pragma omp parallel for
    for (int i=0; i<unvisited_set->count; i++) {

        int node = unvisited_set->vertices[i];

        int start_edge = g->incoming_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[node + 1];
        
        bool into_new_frontier=false;
        for(int neighbor=start_edge;neighbor<end_edge;neighbor++){
            int incoming=g->incoming_edges[neighbor];
            if(distances[incoming]!=NOT_VISITED_MARKER){
                mydistance=distances[incoming]+1;
                while(!__sync_bool_compare_and_swap(new_frontier->vertices+new_frontier->count++,NOT_INIT,node));
                into_new_frontier=true;
                break;
            }
        }
        if(!into_new_frontier){
            while(!__sync_bool_compare_and_swap(new_unvisited_set->vertices+new_unvisited_set->count++,NOT_INIT,node));
        }
    }
    #pragma omp parallel for
    for(int i=0;i<new_frontier->count;i++){
        int node=new_frontier->vertices[i];
        distances[node]=mydistance;
    }
}
```

​		但实验中发现，在更新new_frontier和new_unvisited_set的时候，都需要加上CAS原子操作限制线程的访问，导致性能很差。

​		改进的方法是，因为每轮的frontier到ROOT_NODE的距离都是相同的，并且这个距离就等于进行的轮数，所以用一个**lastdistance**变量表示上一轮得到的frontier到ROOT_NODE的距离，当一个点的distance等于lastdistance就代表它在上一轮算出的frontier上。并且现在unvisited_set就表示除了ROOT_NODE外的所有节点，不再每轮进行更新：

```c++
void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    vertex_set* unvisited_set,
    vertex_set* new_unvisited_set,
    int* distances,
    int lastdistance,
    bool& newadd
    )
{
    int mydistance=0;
    #pragma omp parallel for
    for (int i=0; i<unvisited_set->count; i++) {

        int node = unvisited_set->vertices[i];
        if(distances[node]!=NOT_VISITED_MARKER)continue;
        int start_edge = g->incoming_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[node + 1];
        

        for(int neighbor=start_edge;neighbor<end_edge;neighbor++){
            int incoming=g->incoming_edges[neighbor];
            if(distances[incoming]==lastdistance){                
                distances[node]=lastdistance+1;
                newadd=true;
                break;
            }
        }
    }
    
}
```

​		最后结果如下：



![23adba3a2bb600020c4b3577dcac973](graphs/23adba3a2bb600020c4b3577dcac973.png)