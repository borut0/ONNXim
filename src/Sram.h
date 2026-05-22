#pragma once
#include "Common.h"

class Sram {
 public:
  Sram(SimulationConfig config, const cycle_type& core_cycle, bool accum, uint32_t core_id);

  bool check_hit(addr_type address, int buffer_id);
  bool check_full(int buffer_id);
  bool check_remain(size_t size, int buffer_id);
  bool check_allocated(addr_type address, int buffer_id);

  void cycle();
  void flush(int buffer_id);
  int prefetch(addr_type address, int buffer_id, size_t allocated_size, size_t count);
  void count_up(addr_type, int buffer_id);
  void fill(addr_type address, int buffer_id);
  int get_size() { return _size; }
  int get_current_size(int buffer_id) { return _current_size[buffer_id]; }
  void print_all(int buffer_id);
  // Add these getter methods
  uint64_t get_hit_count() const { return _hit_count; }
  uint64_t get_miss_count() const { return _miss_count; }
  void reset_hit_miss_counters() { _hit_count = 0; _miss_count = 0; }
 private:
  struct SramEntry {
    bool valid;
    addr_type address;
    size_t size;
    size_t remain_req_count;
    cycle_type timestamp;
  };

  int _size;
  int _data_width;
  int _current_size[2];
  bool _accum;

  const cycle_type& _core_cycle;

  robin_hood::unordered_map<addr_type, SramEntry> _cache_table[2];
  // Add these counters at the end of private section
  uint64_t _hit_count = 0;
  uint64_t _miss_count = 0;
};