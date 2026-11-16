import unittest
import numpy as np
from software.quantization import (QParams,SAMPLE,parameters,quantize,dequantize,pack_codes,unpack_codes,integer_dot,required_signed_bits)

class QuantizationTests(unittest.TestCase):
    def test_given_symmetric_example(self):
        p=parameters(SAMPLE,8,'symmetric')
        np.testing.assert_array_equal(quantize(SAMPLE,p),[122,-127,0,65,-124,-32,109,-58,-109,-79])
        self.assertAlmostEqual(p.scale,44.93/127)
    def test_corrected_asymmetric_example(self):
        for method in ['endpoint','affine']:
            p=parameters(SAMPLE,8,method)
            np.testing.assert_array_equal(quantize(SAMPLE,p),[255,0,130,197,3,97,241,71,18,49]);self.assertEqual(p.zero_point,130)
    def test_affine_represents_zero_exactly(self):
        for bits in [2,4,8,16]:
            p=parameters(SAMPLE,bits,'affine');self.assertEqual(float(dequantize([p.zero_point],p)[0]),0)
    def test_endpoint_zero_is_not_integer_affine(self):
        p=parameters(SAMPLE,8,'endpoint');self.assertAlmostEqual(float(dequantize([130],p,np.float64)[0]),.00411764705882689)
    def test_nearest_even_rounding_and_odd_zero_point(self):
        p=QParams(4,'symmetric',1,0,-7,7);np.testing.assert_array_equal(quantize([-2.5,-1.5,.5,1.5,2.5],p),[-2,-2,0,2,2])
        p=QParams(4,'affine',1,3,0,15);np.testing.assert_array_equal(quantize([.5,1.5,2.5],p),[3,5,5])
    def test_all_zero_constant_and_positive_negative_tensors(self):
        for values in [np.zeros(7),np.full(7,3.),np.full(7,-3.),np.array([1.,5.]),np.array([-5.,-1.])]:
            for method in ['symmetric','affine','endpoint']:
                p=parameters(values,8,method);self.assertGreater(p.scale,0);self.assertTrue(np.isfinite(dequantize(quantize(values,p),p)).all())
    def test_invalid_inputs(self):
        for x in [[],[np.nan],[np.inf]]:
            with self.assertRaises(ValueError):parameters(x,8)
        for bits in [1,17,True,4.5]:
            with self.assertRaises(ValueError):parameters([1],bits)
        with self.assertRaises(ValueError):parameters([1],8,'other')
    def test_saturation_and_half_step_error(self):
        for method in ['symmetric','affine','endpoint']:
            x=np.linspace(-2,3,1001);p=parameters(x,4,method);q=quantize(x,p)
            self.assertGreaterEqual(q.min(),p.qmin);self.assertLessEqual(q.max(),p.qmax)
            # Affine rounded zero point can shift an endpoint by up to half a step.
            self.assertLessEqual(np.max(abs(dequantize(q,p,np.float64)-x)),p.scale/2+1e-12)
            np.testing.assert_array_equal(quantize([-1e9,1e9],p),[p.qmin,p.qmax])
    def test_pack_roundtrip_all_codes_and_odd_lengths(self):
        for bits in [2,4,8,16]:
            for signed in [False,True]:
                low=-(1<<(bits-1)) if signed else 0;high=(1<<(bits-1))-1 if signed else (1<<bits)-1
                q=np.arange(low,high+1,dtype=np.int32)
                for a in [q,q[:7],q[:1],q[:0]]:
                    b=pack_codes(a,bits,signed);self.assertEqual(len(b),(len(a)*bits+7)//8);np.testing.assert_array_equal(unpack_codes(b,bits,len(a),signed),a)
    def test_packing_known_endianness(self):
        self.assertEqual(pack_codes([1,2,3],4,False),bytes([0x21,0x03]))
        self.assertEqual(pack_codes([-1,2],16,True),bytes([255,255,2,0]))
        with self.assertRaises(ValueError):pack_codes([4],2,False)
        with self.assertRaises(ValueError):unpack_codes(b'\x01',16,1,False)
    def test_integer_zero_point_expansion(self):
        x=np.array([[5,8,1]]);w=np.array([[6,4,2],[2,3,1]]);zx=3;zw=2;b=np.array([17,-8]);k=x.shape[1]
        expected=x@w.T-zw*x.sum(1,keepdims=True)-zx*w.sum(1)[None]+k*zx*zw+b
        np.testing.assert_array_equal(integer_dot(x,w,zx,zw,b),expected)
        self.assertEqual(required_signed_bits(2**39),41)

if __name__=='__main__':unittest.main()
