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

    namespace {
        using Value = float;
        using Location = Linalg::Vector6D<Value>;
        using Distance = Linalg::Vector3D<Value>;
        using Neighbor = std::pair<Location, Distance>;
        using NeighborToDistance = std::unordered_map<Location, Distance>;
        using Graph = std::unordered_map<Location, NeighborToDistance>;
        using Cost = Distance;
        using LocationToCost = std::unordered_map<Location, Cost>;
        using LocationToParent = std::unordered_map<Location, Location>;
        using LocationsToVisit = std::priority_queue<Location>;
        using LocationsVisited = std::unordered_set<Location>;
        using Path = std::vector<Location>;
    }; // namespace

    static auto make_path(const LocationToParent& location_to_parent, const Location& start, const Location& goal)
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

    static auto make_location_to_empty_parent(const Graph& graph)
    {
        LocationToParent location_to_parent{};
        location_to_parent.reserve(graph.size());
        for (auto const& [location, neighbors] : graph) {
            // start with default initialized parent location
            location_to_parent.emplace_hint(location_to_parent.cend(), location, Location{});
        }
        return location_to_parent;
    }

    static auto make_location_to_infinite_cost(const Graph& graph)
    {
        LocationToCost location_to_cost{};
        location_to_cost.reserve(graph.size());
        for (auto const& [location, neighbors] : graph) {
            // start with infinity initialized cost
            location_to_cost.emplace_hint(location_to_cost.cend(), location, std::numeric_limits<Location>::max());
        }
        return location_to_cost;
    }

    static auto make_empty_locations_to_visit(const Location& start)
    {
        LocationsToVisit locations_to_visit{};
        locations_to_visit.push(start);
        return locations_to_visit;
    }

    static auto make_empty_locations_visited(const Graph& graph)
    {
        LocationsVisited locations_visited{};
        locations_visited.reserve(graph.size());
        return locations_visited;
    }

    [[nodiscard]] auto dijkstra(const Graph& graph, const Location& start, const Location& goal)
    {
        auto locations_to_visit{make_empty_locations_to_visit(start)};
        auto locations_visited{make_empty_locations_visited(graph)};
        auto location_to_parent{make_location_to_empty_parent(graph)};
        auto location_to_cost{make_location_to_infinite_cost(graph)};

        while (!locations_to_visit.empty()) {
            const auto current_location{locations_to_visit.top()};
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

}; // namespace Algorithm

#endif // DIJKSTRA_HPP