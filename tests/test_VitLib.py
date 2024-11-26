import cv2
import numpy as np

from VitLib.VitLib_python import common as common_py
from VitLib.VitLib_cython import common as common_cy

from VitLib.VitLib_python import membrane as membrane_py
from VitLib.VitLib_cython import membrane as membrane_cy

from VitLib.VitLib_python import nucleus as nucleus_py
from VitLib.VitLib_cython import nucleus as nucleus_cy

def test_common():
    img = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    ddac = common_py.detect_deleted_area_candidates(img)
    assert np.all(ddac == common_cy.detect_deleted_area_candidates(img))
    for d in ddac:
        base = common_py.small_area_reduction(img, area_th=d)
        assert np.all(base == common_cy.small_area_reduction_nofix(img, area_th=d))
        assert np.all(base == common_cy.small_area_reduction(img, area_th=d))

    etv = common_py.extract_threshold_values(np.arange(256, dtype=np.uint8).reshape(16, 16))
    assert np.all(etv == common_cy.extract_threshold_values(np.arange(256, dtype=np.uint8).reshape(16, 16)))

def test_membrane():
    img = cv2.imread('tests/img_mem.png', cv2.IMREAD_GRAYSCALE)
    prod = cv2.imread('tests/prod_mem.png', cv2.IMREAD_GRAYSCALE)
    etv = common_cy.extract_threshold_values(img)
    for th in etv[etv>127]:
        img_th = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)[1]
        prod_th = cv2.threshold(prod, th, 255, cv2.THRESH_BINARY)[1]
        prod_th_nwg = membrane_py.NWG(prod_th, symmetric=False)

        base_th_nwg = membrane_py.NWG(img_th, symmetric=False)
        base_th_nwg_sym = membrane_py.NWG(img_th, symmetric=True)

        assert np.all(base_th_nwg == membrane_cy.NWG_nofix(img_th, symmetric=False))
        assert np.all(base_th_nwg_sym == membrane_cy.NWG_nofix(img_th, symmetric=True))

        assert np.all(base_th_nwg == membrane_cy.NWG_old_v0(img_th, symmetric=False))
        assert np.all(base_th_nwg_sym == membrane_cy.NWG_old_v0(img_th, symmetric=True))

        assert np.all(base_th_nwg == membrane_cy.NWG_old_v1(img_th, symmetric=False))
        assert np.all(base_th_nwg_sym == membrane_cy.NWG_old_v1(img_th, symmetric=True))

        assert np.all(base_th_nwg == membrane_cy.NWG(img_th, symmetric=False))
        assert np.all(base_th_nwg_sym == membrane_cy.NWG(img_th, symmetric=True))

        print(f'threshold: {th}, NWG Complete!')

        ddac = common_cy.detect_deleted_area_candidates(prod_th_nwg)
        for del_area in ddac[::len(ddac)//3]:
            img_th_nwg_del = common_cy.small_area_reduction(base_th_nwg, area_th=del_area)

            for radius in range(1, 4):
                base_mlw = membrane_py.modify_line_width(img_th_nwg_del, radius)

                assert np.all(base_mlw == membrane_cy.modify_line_width(img_th_nwg_del, radius))

                base_result = membrane_py.evaluate_membrane_prediction(prod, base_th_nwg, threshold=th, del_area=del_area, radius=radius)
                
                cython_result = membrane_cy.evaluate_membrane_prediction(prod, base_th_nwg, threshold=th, del_area=del_area, radius=radius)
                assert base_result['tip_length'] == cython_result['tip_length'], f'base: {base_result["tip_length"]}, cython: {cython_result["tip_length"]}'
                assert base_result['miss_length'] == cython_result['miss_length'], f'base: {base_result["miss_length"]}, cython: {cython_result["miss_length"]}'
                assert base_result['membrane_length'] == cython_result['membrane_length'], f'base: {base_result["membrane_length"]}, cython: {cython_result["membrane_length"]}'
                
                base_result_nwg = membrane_py.evaluate_membrane_prediction_nwg(prod_th_nwg, base_th_nwg, threshold=th, del_area=del_area, radius=radius)
                assert base_result['tip_length'] == base_result_nwg['tip_length'], f'base: {base_result["tip_length"]}, base_nwg: {base_result_nwg["tip_length"]}'
                assert base_result['miss_length'] == base_result_nwg['miss_length'], f'base: {base_result["miss_length"]}, base_nwg: {base_result_nwg["miss_length"]}'
                assert base_result['membrane_length'] == base_result_nwg['membrane_length'], f'base: {base_result["membrane_length"]}, base_nwg: {base_result_nwg["membrane_length"]}'

                cython_result_nwg = membrane_cy.evaluate_membrane_prediction_nwg(prod_th_nwg, base_th_nwg, threshold=th, del_area=del_area, radius=radius)
                assert base_result['tip_length'] == cython_result_nwg['tip_length'], f'base: {base_result["tip_length"]}, cython_nwg: {cython_result_nwg["tip_length"]}'
                assert base_result['miss_length'] == cython_result_nwg['miss_length'], f'base: {base_result["miss_length"]}, cython_nwg: {cython_result_nwg["miss_length"]}'
                assert base_result['membrane_length'] == cython_result_nwg['membrane_length'], f'base: {base_result["membrane_length"]}, cython_nwg: {cython_result_nwg["membrane_length"]}'

                print(f'threshold: {th}, del_area: {del_area}, radius: {radius}, Complete!')

def test_nucleus():
    img = cv2.imread('tests/img_nuc.png', cv2.IMREAD_GRAYSCALE)
    img_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    prod = cv2.imread('tests/prod_nuc.png', cv2.IMREAD_GRAYSCALE)
    etv = common_cy.extract_threshold_values(prod)
    for th in etv[etv>127]:
        break
        prod_th = cv2.threshold(prod, th, 255, cv2.THRESH_BINARY)[1]
        ddac = common_cy.detect_deleted_area_candidates(prod_th)
        for del_area in ddac[::len(ddac)//3]:
            base_result = nucleus_py.evaluate_nuclear_prediction(prod, img_th, threshold=th, del_area=del_area, eval_mode='inclusion')
            cython_result = nucleus_cy.evaluate_nuclear_prediction(prod, img_th, threshold=th, del_area=del_area, eval_mode='inclusion')

            assert base_result['correct_num'] == cython_result['correct_num'], f'base: {base_result["correct_num"]}, cython: {cython_result["correct_num"]}'
            assert base_result['conformity_bottom'] == cython_result['conformity_bottom'], f'base: {base_result["conformity_bottom"]}, cython: {cython_result["conformity_bottom"]}'
            assert base_result['care_num'] == cython_result['care_num'], f'base: {base_result["care_num"]}, cython: {cython_result["care_num"]}'

            #proximity
            base_result = nucleus_py.evaluate_nuclear_prediction(prod, img_th, threshold=th, del_area=del_area, eval_mode='proximity')
            cython_result = nucleus_cy.evaluate_nuclear_prediction(prod, img_th, threshold=th, del_area=del_area, eval_mode='proximity')

            assert base_result['correct_num'] == cython_result['correct_num'], f'base: {base_result["correct_num"]}, cython: {cython_result["correct_num"]}'
            assert base_result['conformity_bottom'] == cython_result['conformity_bottom'], f'base: {base_result["conformity_bottom"]}, cython: {cython_result["conformity_bottom"]}'
            assert base_result['care_num'] == cython_result['care_num'], f'base: {base_result["care_num"]}, cython: {cython_result["care_num"]}'

            print(f'threshold: {th}, del_area: {del_area}, Complete!')

    results = nucleus_cy.evaluate_nuclear_prediction_range(prod, img_th, min_th=127, max_th=255, step_th=5, min_area=0, max_area=100, step_area=5, eval_mode='proximity', verbose=True)
    print(f'\n\nPrecision: {results[0][2]*100:.3f}, Recall: {results[0][3]*100:.3f}, F-measure: {results[0][4]*100:.3f}, correct_num: {results[0][5]}, conformity_bottom: {results[0][6]}, care_num: {results[0][7]}')

    result = nucleus_py.evaluate_nuclear_prediction(prod, img_th, threshold=129, del_area=0, eval_mode='proximity')
    print(f'Precision: {result["precision"]*100:.3f}, Recall: {result["recall"]*100:.3f}, F-measure: {result["fmeasure"]*100:.3f}, correct_num: {result["correct_num"]}, conformity_bottom: {result["conformity_bottom"]}, care_num: {result["care_num"]}')


if __name__ == "__main__":
    test_nucleus()
