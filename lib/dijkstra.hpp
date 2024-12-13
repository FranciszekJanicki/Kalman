#ifndef DIJKSTRA_HPP
#define DIJKSTRA_HPP

#include "quaternion3d.hpp"
#include "vector3d.hpp"
#include "vector6d.hpp"
#include <algorithm>
#include <queue>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Algorithm {

    template <typename Value>
    struct Dijkstra {
    public:
        using Location = Value;
        using Distance = Value;
        using Neighbor = Location;
        using NeighborToDistance = std::unordered_map<Neighbor, Distance>;
        using Graph = std::unordered_map<Location, NeighborToDistance>;
        using Cost = Distance;
        using LocationToCost = std::unordered_map<Location, Cost>;
        using LocationToParent = std::unordered_map<Location, Location>;
        using LocationsToVisit = std::priority_queue<Location>;
        using LocationsVisited = std::unordered_set<Location>;
        using Path = std::vector<Location>;

        [[nodiscard]] static auto dijkstra(Graph const& graph, Location const& start, Location const& goal)
        {
            auto locations_to_visit{make_empty_locations_to_visit(start)};
            auto locations_visited{make_empty_locations_visited(graph)};
            auto location_to_parent{make_location_to_empty_parent(graph)};
            auto location_to_cost{make_location_to_infinite_cost(graph)};

            while (!locations_to_visit.empty()) {
                auto const current_location{locations_to_visit.top()};
                locations_to_visit.pop();

                if (locations_visited.find(current_location) == locations_visited.cend()) {
                    if (current_location == goal) {
                        break;
                    }
                    auto const& neighbor_location_to_distance{graph.at(current_location)};
                    for (auto const& [neighbor_location, distance] : neighbor_location_to_distance) {
                        if (locations_visited.find(neighbor_location) == locations_visited.cend()) {
                            auto const& old_cost{location_to_cost.at(neighbor_location)};
                            auto const& distance{neighbor_location_to_distance.at(neighbor_location)};
                            auto const& new_cost{location_to_cost.at(current_location) + distance};

                            if (new_cost < old_cost) {
                                location_to_cost.at(neighbor_location) = new_cost;
                                location_to_parent.at(neighbor_location) = current_location;
                                locations_to_visit.push(neighbor_location);
                                locations_visited.insert(current_location);
                            }
                        }
                    }
                }
            }

            return make_path(location_to_parent, start, goal);
        }

    private:
        static auto make_path(LocationToParent const& location_to_parent, Location const& start, Location const& goal)
        {
            Path path{};
            path.reserve(location_to_parent.size());
            auto current_location{goal};
            while (current_location != start) {
                path.push_back(current_location);
                current_location = location_to_parent.at(current_location);
            }
            std::ranges::reverse(path);
            return path;
        }

        static auto make_location_to_empty_parent(Graph const& graph)
        {
            LocationToParent location_to_parent{};
            location_to_parent.reserve(graph.size());
            for (auto const& [location, neighbors] : graph) {
                // start with default initialized parent location
                location_to_parent.emplace_hint(location_to_parent.cend(), location, Location{});
            }
            return location_to_parent;
        }

        static auto make_location_to_infinite_cost(Graph const& graph)
        {
            LocationToCost location_to_cost{};
            location_to_cost.reserve(graph.size());
            for (auto const& [location, neighbors] : graph) {
                // start with infinity initialized cost
                location_to_cost.emplace_hint(location_to_cost.cend(), location, std::numeric_limits<Location>::max());
            }
            return location_to_cost;
        }

        static auto make_empty_locations_to_visit(Location const& start)
        {
            LocationsToVisit locations_to_visit{};
            locations_to_visit.push(start);
            return locations_to_visit;
        }

        static auto make_empty_locations_visited(Graph const& graph)
        {
            LocationsVisited locations_visited{};
            locations_visited.reserve(graph.size());
            return locations_visited;
        }
    };

}; // namespace Algorithm

#endif // DIJKSTRA_HPP