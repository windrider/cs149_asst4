#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <unordered_map>
#include <algorithm>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define NOT_INIT 0

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
    memset(list->vertices,0,sizeof(int)*list->max_vertices);
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
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
        //int mynum=end_edge-start_edge;
        
        // attempt to add all neighbors to the new frontier
        //#pragma omp parallel for if (mynum>1000000000) shared(distances,new_frontier,start_edge,end_edge,g)

        /*#pragma omp parallel if(mynum>10000) shared(distances,new_frontier,start_edge,end_edge,g)
        {
            int thread_count=omp_get_num_threads();
            int thread_id=omp_get_thread_num();
            int step=mynum/thread_count;
            int mystart=start_edge+thread_id*step;
            int myend=(thread_id!=thread_count-1)?mystart+step:end_edge;
        */
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                if (distances[outgoing] == NOT_VISITED_MARKER) {
                    distances[outgoing] = distances[node] + 1;
                    /*
                    int index; 
                    #pragma omp critical
                    index=new_frontier->count++;
                    new_frontier->vertices[index] = outgoing;
                    */

                    while(!__sync_bool_compare_and_swap(new_frontier->vertices+new_frontier->count++,NOT_INIT,outgoing));
                    //#pragma omp atomic
                    //new_frontier->count++;
                    
                }
            }
        //}
    }
}

/*
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
                //#pragma omp critical
                while(!__sync_bool_compare_and_swap(new_frontier->vertices+new_frontier->count++,NOT_INIT,node));
                //new_frontier->vertices[new_frontier->count++]=node;
                into_new_frontier=true;
                break;
            }
        }
        if(!into_new_frontier){
            //while(!__sync_bool_compare_and_swap(new_unvisited_set->vertices+new_unvisited_set->count++,NOT_INIT,node));
            //#pragma omp critical
            new_unvisited_set->vertices[new_unvisited_set->count++]=node;
        }
    }
    #pragma omp parallel for
    for(int i=0;i<new_frontier->count;i++){
        int node=new_frontier->vertices[i];
        distances[node]=mydistance;
    }
}
*/



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
        
        //bool into_new_frontier=false;
        for(int neighbor=start_edge;neighbor<end_edge;neighbor++){
            int incoming=g->incoming_edges[neighbor];
            if(distances[incoming]==lastdistance){
                
                distances[node]=lastdistance+1;
                newadd=true;
                //#pragma omp critical
                //while(!__sync_bool_compare_and_swap(new_frontier->vertices+new_frontier->count++,NOT_INIT,node));
                //new_frontier->vertices[new_frontier->count++]=node;
                //into_new_frontier=true;
                break;
            }
        }
    }
    /*
    #pragma omp parallel for
    for(int i=0;i<new_frontier->count;i++){
        int node=new_frontier->vertices[i];
        distances[node]=mydistance;
    }
    */
    
}


// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    std::unordered_map<int,int> dismap;
    
    vertex_set list1;
    vertex_set list2;
    vertex_set list3;
    vertex_set list4;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set_init(&list3, graph->num_nodes);
    vertex_set_init(&list4, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;
    vertex_set* unvisited_set=&list3;
    vertex_set* new_unvisited_set=&list4;

    // initialize all nodes to NOT_VISITED
    for (int i=1; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
        unvisited_set->vertices[unvisited_set->count++]=i;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    int lastdistance=0;
    bool newadd=true;
    //while(frontier->count!=0 && unvisited_set->count!=0){
    while(newadd){
        vertex_set_clear(new_frontier);
        vertex_set_clear(new_unvisited_set);
        dismap.clear();
        newadd=false;
        bottom_up_step(graph, frontier, new_frontier, unvisited_set,new_unvisited_set,sol->distances,lastdistance,newadd);
        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        lastdistance++;
        /*
        tmp=unvisited_set;
        unvisited_set=new_unvisited_set;
        new_unvisited_set=tmp;
        */


    }
    
    

}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
