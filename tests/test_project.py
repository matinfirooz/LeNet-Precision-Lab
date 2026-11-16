import copy,json,hashlib,unittest
from pathlib import Path
import numpy as np
import torch
from software.model import LeNet,prepare_pixels
from software.data import load_idx,split_indices,FILES
from software.numpy_lenet import forward,prepare_pixels as numpy_pixels
from software.exports import load_export
from software.torch_quantization import load_qdq,convert_native
from software.pruning import global_prune,compact_channels,counts,dense_macs,CompactLeNet
ROOT=Path(__file__).resolve().parents[1]

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.net=LeNet().eval();cls.net.load_state_dict(torch.load(ROOT/'software/lenet_fashion_mnist.pt',weights_only=True)['state_dict']);cls.raw,cls.labels=load_idx(ROOT/'data/fashion','t10k')
    def test_model_parameters_and_macs(self):
        self.assertEqual(sum(p.numel() for p in self.net.parameters()),61706);self.assertEqual(dense_macs(),416520)
        c=CompactLeNet();self.assertEqual(sum(p.numel() for p in c.parameters()),34779);self.assertEqual(dense_macs((4,12,90,63)),231700)
    def test_dataset_checksums_and_disjoint_calibration(self):
        for n,md5 in FILES.items():self.assertEqual(hashlib.md5((ROOT/'data/fashion'/n).read_bytes()).hexdigest(),md5)
        _,labels=load_idx(ROOT/'data/fashion');ti,vi=split_indices(labels);ci=np.load(ROOT/'results/software/calibration_indices.npy')
        self.assertEqual((len(ti),len(vi),len(ci)),(54000,6000,1000));self.assertEqual(len(set(ti)&set(vi)),0);self.assertTrue(set(ci).issubset(set(ti)))
        self.assertEqual(len(self.labels),10000)
    def test_numpy_matches_torch_on_images(self):
        w=dict(np.load(ROOT/'software/trained_weights.npz'));npout=forward(numpy_pixels(self.raw[:32]),w)
        with torch.inference_mode():tout=self.net(prepare_pixels(self.raw[:32])).numpy()
        np.testing.assert_allclose(npout,tout,atol=1e-5,rtol=1e-5);np.testing.assert_array_equal(npout.argmax(1),tout.argmax(1))
    def test_manual_process_had_no_neural_network_imports(self):
        d=json.loads((ROOT/'results/quantization/manual_results.json').read_text());self.assertEqual(d['neural_network_imports'],[]);self.assertEqual(len(d['results']),18);self.assertEqual(d['baseline']['torch_prediction_disagreements'],0)
    def test_every_packed_export_reloads_to_evaluated_weights(self):
        for bits in [2,4,8,16]:
            for method in ['symmetric','affine']:
                w,m=load_export(ROOT/f'hardware/exports/{method}_int{bits}');net=load_qdq(ROOT/f'models/{method}_int{bits}_weights_only.pt').net
                for key,value in net.state_dict().items():np.testing.assert_array_equal(w[key],value.numpy())
                golden=np.load(ROOT/f'hardware/exports/{method}_int{bits}/conv1_integer_golden.npz');self.assertEqual(golden['accumulator'].dtype,np.int64)
    def test_saved_models_reproduce_saved_predictions(self):
        for bits in [2,4,8,16]:
            for method in ['symmetric','affine']:
                for mode in ['weights_only','weights_activations']:
                    m=load_qdq(ROOT/f'models/{method}_int{bits}_{mode}.pt')
                    with torch.inference_mode():pred=m(prepare_pixels(self.raw[:64])).argmax(1).numpy()
                    reference=np.load(ROOT/f'results/quantization/torch_{method}_{bits}_{mode}.npz')['predictions'][:64]
                    np.testing.assert_array_equal(pred,reference)
    def test_native_model_has_quantized_operators_and_reloads(self):
        d=json.loads((ROOT/'results/quantization/pytorch_results.json').read_text());self.assertEqual(len(d['native_operators']),5)
        m=convert_native(self.net,[torch.zeros(1,1,32,32)]);m.load_state_dict(torch.load(ROOT/'models/native_int8.pt',weights_only=True)['state_dict'])
        with torch.inference_mode():pred=m(prepare_pixels(self.raw[:64])).argmax(1).numpy()
        np.testing.assert_array_equal(pred,np.load(ROOT/'results/quantization/native_int8_predictions.npz')['predictions'][:64])
    def test_pruning_masks_survive_optimizer_updates(self):
        m=global_prune(self.net,.75);before={n:b.clone() for n,b in m.named_buffers() if n.endswith('weight_mask')}
        opt=torch.optim.SGD(m.parameters(),lr=.001);loss=m(prepare_pixels(self.raw[:8])).sum();loss.backward();opt.step();m(prepare_pixels(self.raw[:1]))
        self.assertEqual(counts(m)['zero_weights'],46102)
        for n,b in m.named_buffers():
            if n in before:self.assertTrue(torch.equal(b,before[n]))
    def test_compact_equivalence_and_propagated_shapes(self):
        c,s,m=compact_channels(self.net)
        with torch.inference_mode():a=c(prepare_pixels(self.raw[:32]));b=m(prepare_pixels(self.raw[:32]))
        torch.testing.assert_close(a,b,atol=3e-6,rtol=1e-5);self.assertEqual(tuple(c.conv5.weight.shape),(90,12,5,5));self.assertEqual(tuple(c.fc7.weight.shape),(10,63))
    def test_sparse_exports_roundtrip(self):
        for percent in [0,25,50,75,90]:
            path=ROOT/f'hardware/exports/pruned_{percent}_csr';d=json.loads((path/'metadata.json').read_text());state=torch.load(ROOT/f'models/pruned_{percent}.pt',weights_only=True)['state_dict']
            for layer in d['layers']:
                n=layer['layer'];v=np.fromfile(path/f'{n}_values.bin',dtype='<f4');c=np.fromfile(path/f'{n}_columns.bin',dtype='<u2');p=np.fromfile(path/f'{n}_rowptr.bin',dtype='<u4');out=np.zeros((layer['shape'][0],int(np.prod(layer['shape'][1:]))),np.float32)
                for i in range(len(out)):out[i,c[p[i]:p[i+1]]]=v[p[i]:p[i+1]]
                np.testing.assert_array_equal(out.reshape(layer['shape']),state[n+'.weight'].numpy())
    def test_all_result_counts_match_saved_full_test_predictions(self):
        d=json.loads((ROOT/'results/quantization/pytorch_results.json').read_text())
        for row in d['results']:
            if row['method']=='float':filename='float_test_outputs.npz' if row['bits']==32 else 'fp16_predictions.npz';path=ROOT/'results/software'/filename if row['bits']==32 else ROOT/'results/quantization'/filename
            else:path=ROOT/f"results/quantization/torch_{row['method']}_{row['bits']}_{row['mode']}.npz"
            prediction=np.load(path)['predictions'];self.assertEqual(len(prediction),10000);self.assertEqual(int(np.sum(prediction==self.labels)),row['correct'])

    def test_packed_runtime_and_single_image_command(self):
        from software.packed_runtime import PackedLeNet
        import subprocess,sys
        for mode in ['weights_only','weights_activations']:
            m=PackedLeNet(ROOT/'hardware/exports/affine_int8',mode)
            output=m.logits(self.raw[:64])
            reference=np.load(ROOT/f'results/quantization/torch_affine_8_{mode}.npz')
            np.testing.assert_array_equal(output.argmax(1),reference['predictions'][:64])
            np.testing.assert_allclose(m.probabilities(self.raw[:4]).sum(1),np.ones(4),atol=1e-6)
        run=subprocess.run([sys.executable,'scripts/infer_packed.py','--index','0'],cwd=ROOT,text=True,capture_output=True,check=True)
        report=json.loads(run.stdout)
        self.assertEqual(report['neural_network_imports'],[])
        self.assertEqual(report['prediction'],int(np.load(ROOT/'results/quantization/torch_affine_8_weights_only.npz')['predictions'][0]))

    def test_binary_grid_and_payload_corruption_is_rejected(self):
        import tempfile,shutil
        from software.exports import load_export,load_activation_params
        original=ROOT/'hardware/exports/affine_int8'
        for filename in ['weights.bin','biases_fp32.bin','qparams.bin','activation_qparams.bin']:
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'export';shutil.copytree(original,path)
                data=(path/filename).read_bytes()
                if filename=='qparams.bin':
                    changed=bytearray(data);changed[0]^=1;(path/filename).write_bytes(changed)
                else:(path/filename).write_bytes(data[:-1])
                with self.assertRaises(ValueError):
                    if filename=='activation_qparams.bin':load_activation_params(path)
                    else:load_export(path)

    def test_evidence_detects_new_and_changed_computational_inputs(self):
        import tempfile
        from software.evidence import snapshot,validate
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'software').mkdir();(path/'results').mkdir()
            f=path/'software/model.py';f.write_text('value=1\n')
            snapshot(path);self.assertEqual(validate(path)['status'],'passed')
            f.write_text('value=2\n');report=validate(path)
            self.assertEqual(report['status'],'stale');self.assertIn('software/model.py',report['changed_or_missing'])
            f.write_text('value=1\n');(path/'software/extra.py').write_text('value=3\n');report=validate(path)
            self.assertEqual(report['status'],'stale');self.assertEqual(report['unrecorded_files'],['software/extra.py'])

    def test_documentation_uses_changed_experiment_metrics(self):
        import tempfile,shutil
        from software.reporting import generate
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            for folder in ['docs','tests','hardware/exports/affine_int8','results/software','results/quantization','results/pruning']:(path/folder).mkdir(parents=True,exist_ok=True)
            names=['results/software/training_summary.json','results/software/latency.json','results/quantization/manual_results.json','results/quantization/pytorch_results.json','results/quantization/worked_example.json','results/quantization/packed_results.json','results/pruning/pruning_results.json','hardware/exports/affine_int8/metadata.json','tests/test_project.py','tests/test_quantization.py']
            for name in names:shutil.copy2(ROOT/name,path/name)
            def edit(name,change):
                p=path/name;data=json.loads(p.read_text());change(data);p.write_text(json.dumps(data))
            def training(data):data['test'].update(correct=9000,accuracy=.9);data.update(best_epoch=3,epochs_completed=5)
            def precision(data):
                next(v for v in data['results'] if v['name']=='FP32').update(correct=9000,accuracy=.9)
                next(v for v in data['results'] if v['bits']==4 and v['method']=='symmetric' and v['mode']=='weights_only').update(correct=8100,accuracy=.81)
            def pruning(data):data['compact']['after_recovery'].update(correct=8000,accuracy=.8)
            edit('results/software/training_summary.json',training);edit('results/quantization/pytorch_results.json',precision);edit('results/pruning/pruning_results.json',pruning)
            generate(path)
            readme=(path/'README.md').read_text();answers=(path/'COMPLETE_ANSWERS.md').read_text()
            self.assertIn('9000/10,000 = 90.00%',readme);self.assertIn('FP32 scores 90.00%',answers)
            self.assertIn('81.00%',answers);self.assertIn('80.00%',answers)
            self.assertIn('All 5 epochs are completed',answers);self.assertIn('selects epoch 3',answers)

if __name__=='__main__':unittest.main()
