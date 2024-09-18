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
    img = np.array([[   0,   0,   0,   0,   0,   0,   0,   0],
                    [   0, 127, 200, 127,   0, 127, 127,   0],
                    [   0, 255, 255, 255,   0, 255, 255,   0],
                    [   0,   0,   0,   0,   0,   0,   0,   0],
                    [   0,   0, 255, 255, 255,   0,   0,   0],
                    [   0,   0, 255, 255, 255,   0,   0,   0],
                    [   0,   0, 255, 255, 255,   0,   0,   0],
                    [   0,   0, 255,   0,   0,   0,   0,   0]], dtype=np.uint8)
    img = cv2.imread('tests/img.png', cv2.IMREAD_GRAYSCALE)
    prod = cv2.imread('tests/prod.png', cv2.IMREAD_GRAYSCALE)
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

if __name__ == "__main__":
    test_common()
