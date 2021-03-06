class Dic2Obj(object):
    def __init__(self,map):
        self.map = map

    def __setattr__(self, name, value):
        if name == 'map':
             object.__setattr__(self, name, value)
             return;
        print 'set attr called ',name,value
        self.map[name] = value

    def __getattr__(self,name):
        v = self.map[name]
        if isinstance(v,(dict)):
            return Dic2Obj(v)
        if isinstance(v, (list)):
            r = []
            for i in v:
                r.append(Dic2Obj(i))
            return r
        else:
            return self.map[name];

    def __getitem__(self,name):
        return self.map[name]