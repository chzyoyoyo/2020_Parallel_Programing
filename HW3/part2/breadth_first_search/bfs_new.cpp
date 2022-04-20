#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <stdint.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    uint64_t *frontier_bit,
    uint64_t *new_frontier_bit,
    const uint32_t &num_bit)
{
    // #pragma omp parallel for
    uint64_t bias = 1;


    for (int mask = 0; mask < num_bit; mask++)
    {
        if (frontier_bit[mask])
        {
            
            for (int i = 0; i < 64; i++)
            {
                if (frontier_bit[mask] & (1UL << i))
                {
                    int node = (mask<<6) + i;
                    printf("---------------%d\n", node);

                    int start_edge = g->outgoing_starts[node];
                    int end_edge = (node == g->num_nodes - 1)
                                       ? g->num_edges
                                       : g->outgoing_starts[node + 1];

                    // attempt to add all neighbors to the new frontier

                    // int current, next, sum;
                    // #pragma omp parallel for
                    for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
                    {
                        int outgoing = g->outgoing_edges[neighbor];

                        if (distances[outgoing] == NOT_VISITED_MARKER)
                        {
                            distances[outgoing] = distances[node] + 1;

                            int index = new_frontier->count++;
                            new_frontier->vertices[index] = outgoing;
                        }
                    }
                }
            } 
        }
    }
    for (int i = 0; i < num_bit; ++i)
    {
        frontier_bit[i] = 0;
    }
    for (int i = 0; i < new_frontier->count; ++i)
    {
        int offset = i/64;
        int bias = i - ((i/64)*64);
        frontier_bit[offset] |= 1UL << (bias&0x3F);
    }

}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    uint32_t num_bit = (graph->num_nodes + 63) / 64;
    uint64_t *frontier_bit = (uint64_t *)malloc(sizeof(uint64_t) * num_bit);
    uint64_t *new_frontier_bit = (uint64_t *)malloc(sizeof(uint64_t) * num_bit);



    for (int i = 0; i < num_bit; ++i)
    {
        frontier_bit[i] = 0;
        new_frontier_bit[i] = 0;
    }
    // initialize all nodes to NOT_VISITED
    // #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    frontier_bit[ROOT_NODE_ID/64] = 1UL << (ROOT_NODE_ID);
    // frontier_bit[ROOT_NODE_ID/64] |= 1UL << (ROOT_NODE_ID&0x3F);
    printf("frontier_bit[1]: %d\n", frontier_bit[ROOT_NODE_ID/64]);

    while (frontier->count != 0)
    {

// #ifdef VERBOSE
//         double start_time = CycleTimer::currentSeconds();
// #endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances, frontier_bit, new_frontier_bit, num_bit);

// #ifdef VERBOSE
//         double end_time = CycleTimer::currentSeconds();
//         printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
// #endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

// Take one step of "bottom-up" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int depth)
{
    // #pragma omp parallel for
    for (int i = 0; i < g->num_nodes; i++)
    {
        int node = i;

        if (distances[node] == NOT_VISITED_MARKER)
        {
            // printf("1111111\n");
            int start_edge = g->incoming_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                               ? g->num_edges
                               : g->incoming_starts[node + 1];
            // printf("2222222\n");
            // #pragma omp parallel for
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];

                if (distances[incoming] == depth)
                {
                    if (distances[node] == NOT_VISITED_MARKER)
                    {
                        distances[node] = distances[incoming] + 1;
                        int index = new_frontier->count++;
                        new_frontier->vertices[index] = node;
                    }
                    break;
                }
                // }
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
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

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int depth = 0;
    while (frontier->count != 0)
    {

// #ifdef VERBOSE
//         double start_time = CycleTimer::currentSeconds();
// #endif
        vertex_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, depth);

// #ifdef VERBOSE
//         double end_time = CycleTimer::currentSeconds();
//         printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
// #endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        depth++;
    }

}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
//     vertex_set list1;
//     vertex_set list2;
//     vertex_set_init(&list1, graph->num_nodes);
//     vertex_set_init(&list2, graph->num_nodes);

//     vertex_set *frontier = &list1;
//     vertex_set *new_frontier = &list2;

//     int state = 0;

//     // initialize all nodes to NOT_VISITED
//     // #pragma omp parallel for
//     for (int i = 0; i < graph->num_nodes; i++)
//         sol->distances[i] = NOT_VISITED_MARKER;

//     // setup frontier with the root node
//     frontier->vertices[frontier->count++] = ROOT_NODE_ID;
//     sol->distances[ROOT_NODE_ID] = 0;

//     int depth = 0;
//     while (frontier->count != 0)
//     {

// // #ifdef VERBOSE
// //         double start_time = CycleTimer::currentSeconds();
// // #endif

//         vertex_set_clear(new_frontier);
        
//         int mf = 0;
//         int mu = 0;
//         int nf = frontier->count;


//         #pragma omp parallel for
//         for (int i = 0; i < frontier->count; ++i)
//         {
//             int node = frontier->vertices[i];

//             int start_edge = graph->outgoing_starts[node];
//             int end_edge = (node == graph->num_nodes - 1)
//                            ? graph->num_edges
//                            : graph->outgoing_starts[node + 1];
//             #pragma omp atomic
//             mf += (end_edge - start_edge);
//         }
//         #pragma omp parallel for
//         for (int i = 0; i < graph->num_nodes; ++i)
//         {
//             int node = i;

//             if (sol->distances[node] == NOT_VISITED_MARKER)
//             {
//                 int start_edge = graph->incoming_starts[node];
//                 int end_edge = (node == graph->num_nodes - 1)
//                                ? graph->num_edges
//                                : graph->incoming_starts[node + 1];
                
//                 #pragma omp atomic
//                 mu += (end_edge - start_edge);
//             }
//         }
//         if (state == 0)
//         {
//             top_down_step(graph, frontier, new_frontier, sol->distances);
            
//             if (mf > mu/14)
//             {
//                 state = 1;
//             }
//         }
        
//         else
//         {
//             bottom_up_step(graph, frontier, new_frontier, sol->distances, depth);
//             if (nf < graph->num_edges/24)
//             {
//                 state = 0;
//             }
//         }

// // #ifdef VERBOSE
// //         double end_time = CycleTimer::currentSeconds();
// //         printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
// // #endif

//         // swap pointers
//         vertex_set *tmp = frontier;
//         frontier = new_frontier;
//         new_frontier = tmp;
//         depth++;
//     }

}
