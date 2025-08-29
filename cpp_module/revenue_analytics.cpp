#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

class RevenueAnalytics {
public:
    // Fast numerical computations for revenue metrics
    double calculate_adr(const std::vector<double>& revenues, const std::vector<int>& rooms_sold) {
        if (revenues.size() != rooms_sold.size() || revenues.empty()) {
            return 0.0;
        }
        
        double total_revenue = std::accumulate(revenues.begin(), revenues.end(), 0.0);
        int total_rooms = std::accumulate(rooms_sold.begin(), rooms_sold.end(), 0);
        
        return total_rooms > 0 ? total_revenue / total_rooms : 0.0;
    }
    
    // Fast RevPAR calculation
    double calculate_revpar(const std::vector<double>& revenues, const std::vector<int>& available_rooms) {
        if (revenues.size() != available_rooms.size() || revenues.empty()) {
            return 0.0;
        }
        
        double total_revenue = std::accumulate(revenues.begin(), revenues.end(), 0.0);
        int total_rooms = std::accumulate(available_rooms.begin(), available_rooms.end(), 0);
        
        return total_rooms > 0 ? total_revenue / total_rooms : 0.0;
    }
    
    // Fast occupancy rate calculation
    double calculate_occupancy_rate(const std::vector<int>& rooms_sold, const std::vector<int>& available_rooms) {
        if (rooms_sold.size() != available_rooms.size() || rooms_sold.empty()) {
            return 0.0;
        }
        
        int total_sold = std::accumulate(rooms_sold.begin(), rooms_sold.end(), 0);
        int total_available = std::accumulate(available_rooms.begin(), available_rooms.end(), 0);
        
        return total_available > 0 ? (static_cast<double>(total_sold) / total_available) * 100.0 : 0.0;
    }
    
    // Moving average calculation for forecasting
    std::vector<double> moving_average(const std::vector<double>& data, int window_size) {
        std::vector<double> result;
        if (data.size() < static_cast<size_t>(window_size)) {
            return result;
        }
        
        for (size_t i = window_size - 1; i < data.size(); ++i) {
            double sum = 0.0;
            for (int j = 0; j < window_size; ++j) {
                sum += data[i - j];
            }
            result.push_back(sum / window_size);
        }
        
        return result;
    }
    
    // Fast exponential smoothing
    std::vector<double> exponential_smoothing(const std::vector<double>& data, double alpha) {
        std::vector<double> result;
        if (data.empty()) return result;
        
        result.push_back(data[0]);
        
        for (size_t i = 1; i < data.size(); ++i) {
            double smoothed = alpha * data[i] + (1.0 - alpha) * result[i-1];
            result.push_back(smoothed);
        }
        
        return result;
    }
    
    // Statistical calculations
    double calculate_variance(const std::vector<double>& data) {
        if (data.size() < 2) return 0.0;
        
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double variance = 0.0;
        
        for (const auto& value : data) {
            variance += (value - mean) * (value - mean);
        }
        
        return variance / (data.size() - 1);
    }
    
    double calculate_standard_deviation(const std::vector<double>& data) {
        return std::sqrt(calculate_variance(data));
    }
    
    // Correlation coefficient
    double calculate_correlation(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.size() < 2) {
            return 0.0;
        }
        
        double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
        double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
        
        double numerator = 0.0, sum_sq_x = 0.0, sum_sq_y = 0.0;
        
        for (size_t i = 0; i < x.size(); ++i) {
            double dx = x[i] - mean_x;
            double dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        double denominator = std::sqrt(sum_sq_x * sum_sq_y);
        return denominator > 0 ? numerator / denominator : 0.0;
    }
};

namespace py = pybind11;

PYBIND11_MODULE(revenue_analytics_cpp, m) {
    m.doc() = "Fast C++ revenue analytics computations for Streamlit";
    
    py::class_<RevenueAnalytics>(m, "RevenueAnalytics")
        .def(py::init<>())
        .def("calculate_adr", &RevenueAnalytics::calculate_adr,
             "Calculate Average Daily Rate (ADR)")
        .def("calculate_revpar", &RevenueAnalytics::calculate_revpar,
             "Calculate Revenue Per Available Room (RevPAR)")
        .def("calculate_occupancy_rate", &RevenueAnalytics::calculate_occupancy_rate,
             "Calculate occupancy rate percentage")
        .def("moving_average", &RevenueAnalytics::moving_average,
             "Calculate moving average with specified window size")
        .def("exponential_smoothing", &RevenueAnalytics::exponential_smoothing,
             "Apply exponential smoothing with alpha parameter")
        .def("calculate_variance", &RevenueAnalytics::calculate_variance,
             "Calculate variance of data")
        .def("calculate_standard_deviation", &RevenueAnalytics::calculate_standard_deviation,
             "Calculate standard deviation of data")
        .def("calculate_correlation", &RevenueAnalytics::calculate_correlation,
             "Calculate correlation coefficient between two datasets");
}