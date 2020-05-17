from utils import lazy, Singleton


# class AbstractDomain(Singleton): # NO need for abstract domain as it already is a singleton
#
#     def domain(self):
#         raise NotImplementedError
#
#     def domain_botwall(self):
#         raise NotImplementedError
#
#     def domain_topwall(self):
#         raise NotImplementedError
#
#     def domain_barrier(self):
#         raise NotImplementedError
#
#     def domain_barrier_edge(self):
#         raise NotImplementedError


class Barrier:


    def apply_barrier(self):
        raise NotImplementedError

