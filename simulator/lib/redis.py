try:
    import redis
except:
    print("No redis module")

class RedisConn:
    
    def __init__(self, host='localhost', port=6379, db=0):
        self.conn = redis.Redis(host=host, port=port, db=db)

    def is_connected(self):
        try:
            self.conn.ping()
            # print("Successfully connected to redis")
        except (redis.exceptions.ConnectionError, ConnectionRefusedError):
            # print("Redis connection error!")
            return False
        return True

    def reconnect(self, host='localhost', port=6379, db=0):
        self.conn.close()
        self.conn = redis.Redis(host=host, port=port, db=db)

    def get(self, key):
        return self.conn.get(key)

    def set(self, key, value):
        return self.conn.set(key, value)

# generate redis keys to fetch/store data
class RedisKeys:

    base = 'warehouse_visualiser'

    def gen(self, l):
        return ':'.join(l)

    # def gen_metadata_keys(self):
    #     keys = {}
    #     for it in self.metadata:
    #         key = self.gen([self.base, 'metadata', it])
    #         keys[it] = key

    #     return keys

    def gen_timestep_keys(self, timestep, scenario_keys):
        keys = {}
        for it in scenario_keys:
            # key = self.gen([self.base, 'simdata', 't%d'%timestep, it])
            key = self.gen_timestep_key(timestep, it)
            keys[it] = key

        return keys

    def gen_timestep_key(self, timestep, key):
        return self.gen([self.base, 'simdata', 't%d'%timestep, key])

    def gen_metric_timestep_key(self, timestep, metric_id):
        return self.gen_timestep_key(timestep, 'metric:%d'%metric_id)