import numpy as np
import heapq
from scipy.spatial import KDTree
from proxies import compute_proxy
from fixed_cost import fitting_error, compute_cost
import networkx as nx
class Chart:
    def __init__(self, seed, proxy_N, proxy_theta):
        self.seed = seed
        self.proxy_N = proxy_N
        self.proxy_theta = proxy_theta
        self.triangles = [seed]
        self.boundary_edges = set()
        self.queue = []

class DCharts:
    def __init__(self, mesh, F_max=0.2, max_iter=100):
        self.avg_face_area = np.mean(mesh.area_faces) 
        self.mesh = mesh
        self.F_max = F_max
        self.max_iter = max_iter
        self.charts = []
        self.face_adjacency = self._precompute_adjacency()
        self.assigned = np.full(len(mesh.faces), -1, dtype=int)
        self.edge_to_faces = self._precompute_edge_to_faces() 
        self.cost_stats = {'F':[], 'C':[], 'P':[]}
    def _precompute_edge_to_faces(self):
        
        edge_to_faces = {}
        for face_id, face in enumerate(self.mesh.faces):
            for i in range(3):
                v1, v2 = face[i], face[(i+1)%3]
                edge = tuple(sorted([v1, v2]))
                
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_id)
        
        return edge_to_faces
    def compute_boundary_promotion(self, chart, triangle):
        
        l_inner = 0.0
        l_outer = 0.0
        

        tri_verts = self.mesh.faces[triangle]
        candidate_edges = set()
        for i in range(3):
            v1, v2 = tri_verts[i], tri_verts[(i+1)%3]
            candidate_edges.add(tuple(sorted((v1, v2))))
        

        for edge in candidate_edges:
            edge_length = np.linalg.norm(
                self.mesh.vertices[edge[0]] - self.mesh.vertices[edge[1]]
            )
            

            is_shared = False
            if edge in self.edge_to_faces:
                for face_in_chart in self.edge_to_faces[edge]:
                    if face_in_chart in chart.triangles:
                        is_shared = True
                        break
            
            if is_shared:
                l_inner += edge_length
            else:
                l_outer += edge_length
        
        # Handle division by zero
        if l_inner < 1e-6:
            return float('inf') if l_outer > 0 else 1.0
        
        return l_outer / l_inner 
    def _precompute_adjacency(self):
        """Precompute face adjacency list using shared edges"""
        adjacency = [[] for _ in range(len(self.mesh.faces))]
        edge_map = {}
        for fid, face in enumerate(self.mesh.faces):
            for e in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
                edge = tuple(sorted(e))
                if edge in edge_map:
                    neighbor = edge_map[edge]
                    adjacency[fid].append(neighbor)
                    adjacency[neighbor].append(fid)
                else:
                    edge_map[edge] = fid
        return adjacency

    def _farthest_point_sampling(self, num_charts):
        """Correct farthest point sampling with pairwise distances"""
        centroids = self.mesh.triangles_center
        n = len(centroids)
        

        seeds = [np.random.randint(n)]
        distances = np.full(n, np.inf)
        
        for _ in range(1, num_charts):

            new_dists = np.linalg.norm(
                centroids - centroids[seeds[-1]], axis=1)
            distances = np.minimum(distances, new_dists)
            

            next_seed = np.argmax(distances)
            seeds.append(next_seed)
        
        return seeds

    def initialize_seeds(self, num_charts):

        seeds = self._farthest_point_sampling(num_charts)
        for seed in seeds:

            neighbors = set()
            frontier = [seed]
            for _ in range(3):  
                new_frontier = []
                for f in frontier:
                    new_frontier += self.face_adjacency[f]
                neighbors.update(frontier)
                frontier = list(set(new_frontier) - neighbors)
            
            normals = self.mesh.face_normals[list(neighbors)]
            weights = self.mesh.area_faces[list(neighbors)]
            N, theta = compute_proxy(normals, weights)
            
            chart = Chart(seed, N, theta)
            self.charts.append(chart)
            self.assigned[seed] = len(self.charts) - 1
            self._init_chart_queue(chart)

    def _init_chart_queue(self, chart):

        for neighbor in self.face_adjacency[chart.seed]:
            if self.assigned[neighbor] == -1:
                cost = self._calculate_cost(neighbor, chart)
                heapq.heappush(chart.queue, (cost, neighbor))

    def compute_compactness(self, chart, triangle):
 


        G = nx.Graph()
        chart_faces = chart.triangles + [triangle]  

        for f in chart_faces:
            G.add_node(f)
            for neighbor in self.face_adjacency[f]:
                if neighbor in chart_faces:

                    weight = np.linalg.norm(
                        self.mesh.triangles_center[f] - 
                        self.mesh.triangles_center[neighbor]
                    )
                    G.add_edge(f, neighbor, weight=weight)

        path_length = nx.shortest_path_length(G, 
                                             source=chart.seed,
                                             target=triangle,
                                             weight='weight')


        chart_area = np.sum(self.mesh.area_faces[chart.triangles]) 
        normalized_area = max(chart_area, 0.1 * self.avg_face_area)
        C = np.pi * (path_length ** 2) / normalized_area

        return C



    def _calculate_cost(self, triangle, chart):
        """Improved cost calculation with normalization"""

        raw_F = fitting_error(self.mesh.face_normals[triangle],
                      chart.proxy_N, chart.proxy_theta)

        if raw_F > self.F_max:
            return float('inf')


        F = raw_F / 4.0



        C = self.compute_compactness(chart, triangle)

        current_face = set(self.mesh.faces[triangle])
        shared_edges = 0
        for cf in chart.triangles:
            chart_face = set(self.mesh.faces[cf])
            if len(current_face & chart_face) >= 2:  
                shared_edges += 1
  
        P = self.compute_boundary_promotion(chart, triangle)


        if not hasattr(self, 'cost_stats'):
            self.cost_stats = {'F':[], 'C':[], 'P':[]}
        self.cost_stats['F'].append(F)
        self.cost_stats['C'].append(C)
        self.cost_stats['P'].append(P)

        if C > 100 or F > 1.0:
            print(f"EXTREME t{triangle} -> F={F:.3f}, C={C:.3f}, P={P:.3f}")




        if np.isinf(F) or np.isinf(C) or np.isinf(P):
            return float('inf')
        
        
            
        return compute_cost(F, C, P)

    def _grow_chart(self, chart):
        processed = set()  
        while chart.queue:
            cost, triangle = heapq.heappop(chart.queue)
            
            if triangle in processed:
                continue
            processed.add(triangle)
            
  
            if len(chart.triangles) > 1:  
                current_face = set(self.mesh.faces[triangle])
                is_connected = False
                for cf in chart.triangles:
                    chart_face = set(self.mesh.faces[cf])
                    if len(current_face & chart_face) >= 2: 
                        is_connected = True
                        break
                if not is_connected:
                    continue  
            

            current_chart = self.assigned[triangle]
            if current_chart != -1:

                if current_chart != self.charts.index(chart):
                    current_cost = self._calculate_cost(triangle, self.charts[current_chart])
                    if cost < current_cost * 0.9:  

                        if self._maintains_connectivity(self.charts[current_chart], triangle):
                            self.charts[current_chart].triangles.remove(triangle)
                        else:
                            continue
                    else:
                        continue
            

            self.assigned[triangle] = self.charts.index(chart)
            chart.triangles.append(triangle)
            

            new_normals = self.mesh.face_normals[chart.triangles]
            new_weights = self.mesh.area_faces[chart.triangles]
            chart.proxy_N, chart.proxy_theta = compute_proxy(new_normals, new_weights)
            

            for neighbor in self.face_adjacency[triangle]:
                if neighbor not in processed:
                    new_cost = self._calculate_cost(neighbor, chart)
                    heapq.heappush(chart.queue, (new_cost, neighbor))

    def _maintains_connectivity(self, chart, face_to_remove):
        """Check if removing a face would break chart connectivity"""
        if len(chart.triangles) <= 2:  
            return False
            
        remaining = set(chart.triangles) - {face_to_remove}
        if not remaining:
            return False
            

        connected = {next(iter(remaining))}
        frontier = [next(iter(remaining))]
        
        while frontier:
            current = frontier.pop()
            for neighbor in self.face_adjacency[current]:
                if neighbor in remaining and neighbor not in connected:
                    connected.add(neighbor)
                    frontier.append(neighbor)
        

        return len(connected) == len(remaining)

                    
    def lloyd_iterations(self):


        for iteration in range(self.max_iter):

            self.assigned[:] = -1


            for chart_index, chart in enumerate(self.charts):
                chart.triangles = [chart.seed]
                chart.queue = []
                self.assigned[chart.seed] = chart_index
                self._init_chart_queue(chart)  


            all_charts_active = True
            while all_charts_active:
                all_charts_active = False
                for chart in self.charts:
                    if chart.queue:

                        cost, triangle = heapq.heappop(chart.queue)
                        

                        if self.assigned[triangle] != -1:
                            continue
                            

                        if len(chart.triangles) > 1:
                            current_face = set(self.mesh.faces[triangle])
                            is_connected = False
                            for cf in chart.triangles:
                                chart_face = set(self.mesh.faces[cf])
                                if len(current_face & chart_face) >= 2:  
                                    is_connected = True
                                    break
                            if not is_connected:
                                continue
                        

                        self.assigned[triangle] = self.charts.index(chart)
                        chart.triangles.append(triangle)
                        

                        new_normals = self.mesh.face_normals[chart.triangles]
                        new_weights = self.mesh.area_faces[chart.triangles]
                        chart.proxy_N, chart.proxy_theta = compute_proxy(new_normals, new_weights)
                        
                        # Add neighbors to queue
                        for neighbor in self.face_adjacency[triangle]:
                            if self.assigned[neighbor] == -1:
                                new_cost = self._calculate_cost(neighbor, chart)
                                heapq.heappush(chart.queue, (new_cost, neighbor))
                        
                        all_charts_active = True
            

            sizes = [len(c.triangles) for c in self.charts]
            print(f"Iter {iteration}: Chart sizes {sizes}")
            

            print(f"\nIter {iteration} Cost Stats:")
            for k in ['F', 'C', 'P']:
                vals = self.cost_stats[k]
                if vals:  
                    print(f"{k}: min={min(vals):.3f}, max={max(vals):.3f}, avg={np.mean(vals):.3f}")
                else:
                    print(f"{k}: (no data this iteration)")

            self.cost_stats = {'F':[], 'C':[], 'P':[]}


            for chart in self.charts:
                if chart.triangles:
                    normals = self.mesh.face_normals[chart.triangles]
                    weights = self.mesh.area_faces[chart.triangles]
                    chart.proxy_N, chart.proxy_theta = compute_proxy(normals, weights)


            changed = False
            for chart in self.charts:
                if chart.triangles:
                    centroids = self.mesh.triangles_center[chart.triangles]
                    mean_pos = np.mean(centroids, axis=0)

                    new_seed_idx = np.argmin(np.linalg.norm(centroids - mean_pos, axis=1))
                    new_seed = chart.triangles[new_seed_idx]
                    if new_seed != chart.seed:
                        chart.seed = new_seed
                        changed = True


            if not changed:
                print(f"Converged after {iteration + 1} iterations.")
                break


    def fill_holes(self):
        """hole filling to handle disconnected faces"""
        unassigned = np.where(self.assigned == -1)[0]
        

        for face in unassigned:

            min_cost = float('inf')
            best_chart = None
            

            for neighbor in self.face_adjacency[face]:
                if self.assigned[neighbor] != -1:
                    chart = self.charts[self.assigned[neighbor]]
                    try:
                        cost = self._calculate_cost(face, chart)
                        if cost < min_cost:
                            min_cost = cost
                            best_chart = chart
                    except Exception:
                        continue
            
            if best_chart:
                self.assigned[face] = self.charts.index(best_chart)
                best_chart.triangles.append(face)
        
        # Second pass: For any remaining unassigned faces, use nearest chart
        still_unassigned = np.where(self.assigned == -1)[0]
        if len(still_unassigned) > 0:
            print(f"Second pass hole filling: {len(still_unassigned)} faces still unassigned")
            # Use face centroids for distance calculation
            centroids = self.mesh.triangles_center
            chart_centroids = [np.mean(centroids[chart.triangles], axis=0) for chart in self.charts]
            
            for face in still_unassigned:
                # Find nearest chart by centroid distance
                face_centroid = centroids[face]
                distances = [np.linalg.norm(face_centroid - chart_centroid) for chart_centroid in chart_centroids]
                nearest_chart_idx = np.argmin(distances)
                
                # Assign to nearest chart
                self.assigned[face] = nearest_chart_idx
                self.charts[nearest_chart_idx].triangles.append(face)
            
            print(f"After second pass: {np.sum(self.assigned == -1)} faces remain unassigned")
        
        # Third pass: Check for any isolated faces within charts and reassign them
        for chart_idx, chart in enumerate(self.charts):

            if len(chart.triangles) < 5:
                continue
                

            G = nx.Graph()
            for face in chart.triangles:
                G.add_node(face)
                for neighbor in self.face_adjacency[face]:
                    if neighbor in chart.triangles:
                        G.add_edge(face, neighbor)
            

            components = list(nx.connected_components(G))
            

            if len(components) > 1:
                print(f"Chart {chart_idx} has {len(components)} disconnected components")
                largest_component = max(components, key=len)
                

                for component in components:
                    if component != largest_component:
                        for face in component:
                            # Find best neighboring chart
                            best_neighbor_chart = None
                            min_neighbor_cost = float('inf')
                            
                            for neighbor in self.face_adjacency[face]:
                                if self.assigned[neighbor] != chart_idx:
                                    neighbor_chart_idx = self.assigned[neighbor]
                                    if neighbor_chart_idx != -1:
                                        neighbor_chart = self.charts[neighbor_chart_idx]
                                        try:
                                            cost = self._calculate_cost(face, neighbor_chart)
                                            if cost < min_neighbor_cost:
                                                min_neighbor_cost = cost
                                                best_neighbor_chart = neighbor_chart
                                        except Exception:
                                            continue
                            

                            if best_neighbor_chart:
                                chart.triangles.remove(face)
                                best_neighbor_chart_idx = self.charts.index(best_neighbor_chart)
                                self.assigned[face] = best_neighbor_chart_idx
                                best_neighbor_chart.triangles.append(face)


    def segment(self, num_charts=10):
        self.initialize_seeds(num_charts)
        self.lloyd_iterations()
        self.fill_holes()
        return [chart.triangles for chart in self.charts]
