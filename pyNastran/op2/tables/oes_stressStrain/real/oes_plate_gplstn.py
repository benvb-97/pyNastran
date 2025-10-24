"""
defines:
 - RealNonlinearPlateArray

"""
from math import isnan
from itertools import count, cycle

import numpy as np

from pyNastran.utils.numpy_utils import integer_types
from pyNastran.op2.result_objects.op2_objects import get_times_dtype
from pyNastran.op2.tables.oes_stressStrain.real.oes_objects import OES_Object
from pyNastran.f06.f06_formatting import _eigenvalue_header, write_float_11e, write_float_13e


class RealGPlStnPlateArray(OES_Object):
    """Generalized plane strain elements array"""
    def __init__(self, data_code, is_sort1, isubcase, dt):
        OES_Object.__init__(self, data_code, isubcase, apply_data_code=True)
        #self.code = [self.format_code, self.sort_code, self.s_code]

        #self.ntimes = 0  # or frequency/mode
        #self.ntotal = 0
        self.ielement = 0
        self.nelements = 0  # result specific
        self.nnodes = None

    @property
    def is_real(self) -> bool:
        return True

    @property
    def is_complex(self) -> bool:
        return False

    @property
    def nnodes_per_element(self) -> int:
        if self.element_type in [328, 330]:  # GPLSTN3, GPLSTN6
            nnodes_per_element = 3
        elif self.element_type in [329, 331]:  # GPLSTN4, GPLSTN8
            nnodes_per_element = 4
        else:
            raise NotImplementedError('name=%r type=%s' % (self.element_name, self.element_type))
        return nnodes_per_element

    def _reset_indices(self) -> None:
        self.itotal = 0
        self.ielement = 0

    @property
    def is_stress(self):
        return True

    def get_headers(self) -> list[str]:
        headers = []
        return headers

    def build(self):
        """sizes the vectorized attributes of the RealNonlinearPlateArray"""
        # print("self.ielement = %s" % self.ielement)
        #print('ntimes=%s nelements=%s ntotal=%s' % (self.ntimes, self.nelements, self.ntotal))
        assert self.ntimes > 0, 'ntimes=%s' % self.ntimes
        assert self.nelements > 0, 'nelements=%s' % self.nelements
        assert self.ntotal > 0, 'ntotal=%s' % self.ntotal

        if self.nelements % self.ntimes != 0:
            msg = 'nelements=%s ntimes=%s nelements/ntimes=%s'  % (
                self.nelements, self.ntimes, self.nelements / float(self.ntimes))
            #return
            raise RuntimeError(msg)

        if self.is_sort1:
            self.nelements //= self.ntimes
        self.nnodes = self.nelements * self.nnodes_per_element
        self.itime = 0
        self.ielement = 0
        self.itotal = 0
        #self.ntimes = 0
        #self.nelements = 0

        if self.is_sort1:
            ntimes = self.ntimes
            ntotal = self.ntotal
            nelements = self.nelements
        else:
            nelements = self.ntimes
            ntimes = self.nelements // self.ntimes // 2
            #print("RealNonlinearPlateArray: name=%s type=%s nnodes_per_element=%s ntimes=%s nelements=%s ntotal=%s" % (
                #self.element_name, self.element_type, nnodes_per_element, self.ntimes, self.nelements, self.ntotal))
            # ntotal = self.ntotal // ntimes
            ntotal = nelements * self.nnodes_per_element
            self.ntimes = ntimes
            self.nelements = nelements
            #print("-> ntimes=%s nelements=%s ntotal=%s" % (
                #ntimes, nelements, ntotal))

            assert nelements > 0, nelements
        assert ntotal > 0, ntotal

            #print("***name=%s type=%s nnodes_per_element=%s ntimes=%s nelements=%s ntotal=%s" % (
                #self.element_name, self.element_type, nnodes_per_element, self.ntimes, self.nelements, self.ntotal))
        dtype, idtype, fdtype = get_times_dtype(self.nonlinear_factor, self.size, self.analysis_fmt)
        self._times = np.zeros(ntimes, dtype=self.analysis_fmt)
        self.element_node = np.zeros((self.nnodes, 2), dtype=idtype)

        #[oxx, oyy, ozz, oxy, oyz, ozx, ovm]
        self.data = np.zeros((ntimes, self.nnodes, 7), dtype=fdtype)
        self.thetas = np.zeros(nelements, dtype=idtype)  # material orientation angles

    def add_new_eid_sort1(self, dt, eid, etype, theta, ex, ey, ez, exy, eyz, ezx, evm):
        self.element[self.ielement] = eid
        self.ielement += 1
        self.add_sort1(dt, eid, etype, theta, ex, ey, ez, exy, eyz, ezx, evm)

    def add_new_eid_sort2(self, dt, eid, etype, theta, ex, ey, ez, exy, eyz, ezx, evm):
        ielement = self.itime
        self.element[ielement] = eid
        #self.ielement += 1
        self.add_sort2(dt, eid, etype, theta, ex, ey, ez, exy, eyz, ezx, evm)

    def add_sort1(self, dt, eid, etype, theta, ex, ey, ez, exy, eyz, ezx, evm):
        """unvectorized method for adding SORT1 transient data"""
        assert self.sort_method == 1, self
        assert isinstance(eid, integer_types) and eid > 0, 'dt=%s eid=%s' % (dt, eid)

        self._times[self.itime] = dt
        #if self.ielement == 10:
            #print(self.element_node[:10, :])
            #raise RuntimeError()
        #[fiber_dist, oxx, oyy, ozz, txy, es, eps, ecs, exx, eyy, ezz, etxy]
        assert eid == self.element[self.ielement - 1], 'eid=%s self.element[i-1]=%s' % (eid, self.element[self.ielement - 1])
        self.data[self.itime, self.itotal, :] = [ex, ey, ez, exy, eyz, ezx, evm]
        self.thetas[self.ielement] = theta
        self.itotal += 1

    def add_sort2(self, dt, eid, etype, theta, ex, ey, ez, exy, eyz, ezx, evm):
        """unvectorized method for adding SORT2 transient data"""
        assert self.sort_method == 2, self
        assert isinstance(eid, integer_types) and eid > 0, 'dt=%s eid=%s' % (dt, eid)

        ntimes = len(self._times)
        nelement = len(self.element)
        ntotal = self.data.shape[1]

        ilayer = self.itotal % 2
        ielement = self.itime

        itotal = self.itotal % ntotal
        itime = self.itotal // nelement // 2
        #itotal = self.ielement

        #try:
        self._times[itime] = dt
        #except:
            #pass
        utimes = np.round(np.unique(self._times), 3) # .tolist()
        #print('[%s]' % (', '.join('%g' % val for val in utimes)), '; n=%s' % len(utimes))

        #[fiber_dist, oxx, oyy, ozz, txy, es, eps, ecs, exx, eyy, ezz, etxy]
        #print(f'RealNonlinearPlateArray: itime={itime}/{ntimes} ielement={ielement}/{nelement} ilayer={ilayer} itotal={itotal}/{ntotal} -> dt={dt:<-5g} eid={eid}')
        #assert eid == self.element[ielement - 1], 'eid=%s self.element[i-1]=%s' % (eid, self.element[self.ielement - 1])
        self.data[itime, itotal, :] = [ex, ey, ez, exy, eyz, ezx, evm]
        self.thetas[self.ielement] = theta
        self.itotal += 1

    def __eq__(self, table):  # pragma: no cover
        self._eq_header(table)
        assert self.is_sort1 == table.is_sort1
        if not np.array_equal(self.data, table.data):
            msg = 'table_name=%r class_name=%s\n' % (self.table_name, self.__class__.__name__)
            msg += '%s\n' % str(self.code_information())
            i = 0
            for itime in range(self.ntimes):
                for ielem, eid in enumerate(self.element):
                    t1 = self.data[itime, ielem, :]
                    t2 = table.data[itime, ielem, :]

                    # TODO: this name order is wrong
                    #[fiber_dist, oxx, oyy, ozz, txy, es, eps, ecs, exx, eyy, ezz, etxy]
                    (theta1, ex1, ey1, ez1, exy1, eyz1, ezx1, evm1) = t1
                    (theta2, ex2, ey2, ez2, exy2, eyz2, ezx2, evm2) = t2

                    # vm stress can be NaN for some reason...
                    if not np.array_equal(t1, t2):
                        eid_spaces = ' ' * (len(str(eid)))
                        msg += (
                            # eid   fd  ox  oy  oz  txy ex  ey  ez  exy es  eps ecs1
                            '%s    (%s, %s, %s, %s, %s, %s, %s, %s)\n'
                            '%s    (%s, %s, %s, %s, %s, %s, %s, %s)\n' % (
                                eid,
                                theta1, ex1, ey1, ez1, exy1, eyz1, ezx1, evm1,
                                eid_spaces,
                                theta2, ex2, ey2, ez2, exy2, eyz2, ezx2, evm2))
                        i += 1
                        if i > 10:
                            print(msg)
                            raise ValueError(msg)
                #print(msg)
                if i > 0:
                    raise ValueError(msg)
        return True

    def get_stats(self, short: bool=False) -> list[str]:
        if not self.is_built:
            return [
                f'<{self.__class__.__name__}>; table_name={self.table_name!r}\n',
                f'  ntimes: {self.ntimes:d}\n',
                f'  ntotal: {self.ntotal:d}\n',
            ]

        nelements = self.nelements
        ntimes = self.ntimes
        nnodes = self.nnodes
        ntotal = self.ntotal
        nelements = self.ntotal // self.nnodes // 2

        msg = []
        if self.nonlinear_factor not in (None, np.nan):  # transient
            msgi = '  type=%s ntimes=%i nelements=%i nnodes_per_element=%i ntotal=%i, table_name=%s\n' % (
                self.__class__.__name__, ntimes, nelements, nnodes, ntotal, self.table_name_str)
            ntimes_word = 'ntimes'
        else:
            msgi = '  type=%s nelements=%i nnodes_per_element=%i ntotal=%i\n' % (
                self.__class__.__name__, nelements, nnodes, ntotal)
            ntimes_word = '1'
        msg.append(msgi)
        headers = self.get_headers()
        n = len(headers)
        msg.append('  data: [%s, ntotal, %i] where %i=[%s]\n' % (ntimes_word, n, n,
                                                                 str(', '.join(headers))))
        msg.append('  data.shape=%s\n' % str(self.data.shape))
        msg.append(f'  element type: {self.element_name}-{self.element_type}\n')
        msg += self.get_data_code()
        return msg

    def write_f06(self, f06_file, header=None, page_stamp='PAGE %s', page_num=1,
                  is_mag_phase=False, is_sort1=True):
        if header is None:
            header = []
        #msg, nnodes, cen = _get_plate_msg(self)

        element_type_mapper = {
            328: "3", 329: "4", 330: "6", 331: "8",
        }
        if self.element_type in element_type_mapper:
            msg = [
                f'                   S T R E S S E S   I N    T R I A N G U L A R   E L E M E N T S    ( G P L S T N {element_type_mapper[self.element_type]} )\n'
                ' \n'
                '    ELEMENT   THETA  GRID/  POINT   ENG MEASURE           C O M P O N E N T S   I N   M A T E R I A L   C. S.         VON MISES\n'
                '    ID               GAUSS  ID       NORMAL-X      NORMAL-Y      NORMAL-Z      SHEAR-XY      SHEAR-YZ      SHEAR-ZX    EQUIV STRESS\n'
            ]
        else:  # pragma: no cover
            raise NotImplementedError('element_name=%s self.element_type=%s' % (self.element_name, self.element_type))

        # write the f06
        ntimes = self.data.shape[0]
        eids = self.element_node[:, 0]
        nids = self.element_node[:, 1]

        #cen_word = 'CEN/%i' % nnodes
        for itime in range(ntimes):
            dt = self._times[itime]
            header = _eigenvalue_header(self, header, itime, ntimes, dt)
            f06_file.write(''.join(header + msg))

            #print("self.data.shape=%s itime=%s ieids=%s" % (str(self.data.shape), itime, str(ieids)))

            #[ex, ey, ez, exy, eyz, ezx, evm]
            ex = self.data[itime, :, 0]
            ey = self.data[itime, :, 1]
            ez = self.data[itime, :, 2]
            exy = self.data[itime, :, 3]
            eyz = self.data[itime, :, 4]
            ezx = self.data[itime, :, 5]
            evm = self.data[itime, :, 6]

            for (i, eid, nid, theta, exi, eyi, ezi, exyi, eyzi, ezxi, evmi) in zip(cycle(range(self.nnodes_per_element)), eids, nids, self.thetas, ex, ey, ez, exy, eyz, ezx, evm):
                """
                     ELEMENT   THETA  GRID/  POINT   ENG MEASURE           C O M P O N E N T S   I N   M A T E R I A L   C. S.         VON MISES/   
                     ID               GAUSS  ID       NORMAL-X      NORMAL-Y      NORMAL-Z      SHEAR-XY      SHEAR-YZ      SHEAR-ZX    EQUIV STRESS
                           1   0.0    GRID
                                                 6  6.620830E+03  5.604940E+01  4.486742E+01  5.765791E+01  0.0           0.0           6.571137E+03
                                                 2  6.620830E+03  5.604940E+01  4.486742E+01  5.765791E+01  0.0           0.0           6.571137E+03
                                                 1  6.620830E+03  5.604940E+01  4.486742E+01  5.765791E+01  0.0           0.0           6.571137E+03
                           2   0.0    GRID
                                                 6  6.711007E+03  4.434020E+01  7.870317E+01  4.434020E+01  0.0           0.0           6.649995E+03
                                                 1  6.711007E+03  4.434020E+01  7.870317E+01  4.434020E+01  0.0           0.0           6.649995E+03
                                                 4  6.711007E+03  4.434020E+01  7.870317E+01  4.434020E+01  0.0           0.0           6.649995E+03
                """
                if i == 0:
                    f06_file.write(
                        '   %8i  %-13s  GRID\n' % (eid, write_float_13e(theta)))

                f06_file.write(
                    '                          %8i  %-13s  %-13s  %-13s  %-13s  %-13s  %-13s  %s\n' % (
                        nid,
                        write_float_13e(exi), write_float_13e(eyi), write_float_13e(ezi),
                        write_float_13e(exyi), write_float_13e(eyzi), write_float_13e(ezxi), write_float_13e(evmi),
                    ))

            f06_file.write(page_stamp % page_num)
            page_num += 1
        return page_num - 1
