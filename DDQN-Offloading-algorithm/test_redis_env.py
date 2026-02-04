#!/usr/bin/env python3
"""
Test Redis ML Environment Integration
"""
import sys
import redis
import psycopg2

def test_redis_connection():
    """Test Redis connection."""
    print("1. Testing Redis connection...")
    try:
        r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        if r.ping():
            print("   ✓ Redis connected successfully")
            return True
        else:
            print("   ✗ Redis ping failed")
            return False
    except Exception as e:
        print(f"   ✗ Redis connection failed: {e}")
        return False

def test_redis_python_library():
    """Test redis-py library is installed."""
    print("2. Testing redis-py library...")
    try:
        import redis
        print(f"   ✓ redis-py version: {redis.__version__}")
        return True
    except ImportError:
        print("   ✗ redis-py not installed. Run: pip install redis>=4.5.0")
        return False

def test_redis_data():
    """Test if Redis has simulation data."""
    print("3. Testing Redis data availability...")
    try:
        r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        
        # Check for vehicles
        vehicle_keys = r.keys("vehicle:*:state")
        print(f"   - Found {len(vehicle_keys)} vehicle states")
        
        # Check for service vehicles
        service_count = r.zcard("service_vehicles:available")
        print(f"   - Found {service_count} service vehicles in sorted set")
        
        # Check for tasks
        task_keys = r.keys("task:*:state")
        print(f"   - Found {len(task_keys)} task states")
        
        # Check for RSU
        rsu_keys = r.keys("rsu:*:resources")
        print(f"   - Found {len(rsu_keys)} RSU resource states")
        
        if vehicle_keys or task_keys or rsu_keys:
            print("   ✓ Redis has simulation data")
            return True
        else:
            print("   ⚠ Redis is empty (simulation may not be running)")
            return True  # Not an error, just no data yet
            
    except Exception as e:
        print(f"   ✗ Redis data check failed: {e}")
        return False

def test_sample_query():
    """Test a sample Redis query."""
    print("4. Testing sample Redis query...")
    try:
        r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        
        # Try to get a vehicle
        vehicle_keys = r.keys("vehicle:*:state")
        if vehicle_keys:
            sample_key = vehicle_keys[0]
            state = r.hgetall(sample_key)
            print(f"   ✓ Retrieved state for {sample_key.split(':')[1]}")
            print(f"     - CPU available: {state.get('cpu_available', 'N/A')}")
            print(f"     - Queue length: {state.get('queue_length', 'N/A')}")
            return True
        else:
            print("   ⚠ No vehicles in Redis to query")
            return True
            
    except Exception as e:
        print(f"   ✗ Sample query failed: {e}")
        return False

def test_performance():
    """Test Redis query performance."""
    print("5. Testing Redis query performance...")
    try:
        import time
        r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        
        # Test single query
        vehicle_keys = r.keys("vehicle:*:state")
        if vehicle_keys:
            key = vehicle_keys[0]
            
            # Warm up
            r.hgetall(key)
            
            # Measure
            iterations = 100
            start = time.time()
            for _ in range(iterations):
                r.hgetall(key)
            elapsed = (time.time() - start) * 1000  # Convert to ms
            avg_latency = elapsed / iterations
            
            print(f"   ✓ Average query latency: {avg_latency:.3f}ms ({iterations} iterations)")
            
            if avg_latency < 1.0:
                print(f"   ✓ Excellent performance (< 1ms)")
            elif avg_latency < 5.0:
                print(f"   ✓ Good performance (< 5ms)")
            else:
                print(f"   ⚠ Slow performance (> 5ms) - check Redis configuration")
            
            return True
        else:
            print("   ⚠ No data to benchmark")
            return True
            
    except Exception as e:
        print(f"   ✗ Performance test failed: {e}")
        return False

def test_environment_import():
    """Test if IoVRedisEnv can be imported."""
    print("6. Testing IoVRedisEnv import...")
    try:
        sys.path.insert(0, '.')
        from src.environment import IoVRedisEnv
        print("   ✓ IoVRedisEnv imported successfully")
        return True
    except ImportError as e:
        print(f"   ✗ Failed to import IoVRedisEnv: {e}")
        return False

def main():
    print("=" * 60)
    print("Redis ML Environment Integration Test")
    print("=" * 60)
    print()
    
    results = []
    results.append(("Redis Connection", test_redis_connection()))
    results.append(("Redis Python Library", test_redis_python_library()))
    results.append(("Redis Data", test_redis_data()))
    results.append(("Sample Query", test_sample_query()))
    results.append(("Performance", test_performance()))
    results.append(("Environment Import", test_environment_import()))
    
    print()
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print()
    
    if all(r[1] for r in results):
        print("✓ All tests passed! Redis ML integration is ready.")
        print()
        print("Next steps:")
        print("  1. Start simulation: ./complex-network")
        print("  2. Run ML with Redis: python main.py --env redis")
        print("  3. Monitor Redis: redis-cli monitor")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print()
        print("Common fixes:")
        print("  - Install Redis: sudo apt-get install redis-server")
        print("  - Install redis-py: pip install redis>=4.5.0")
        print("  - Start Redis: sudo systemctl start redis")
        return 1

if __name__ == "__main__":
    sys.exit(main())
