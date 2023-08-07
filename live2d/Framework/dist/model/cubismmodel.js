"use strict";
/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Live2DCubismFramework = exports.CubismModel = exports.DrawableCullingData = exports.PartColorData = exports.DrawableColorData = void 0;
var live2dcubismframework_1 = require("../live2dcubismframework");
var cubismrenderer_1 = require("../rendering/cubismrenderer");
var csmmap_1 = require("../type/csmmap");
var csmvector_1 = require("../type/csmvector");
var cubismdebug_1 = require("../utils/cubismdebug");
/**
 * SDK側から与えられたDrawableの乗算色・スクリーン色上書きフラグと
 * その色を保持する構造体
 */
var DrawableColorData = /** @class */ (function () {
    function DrawableColorData(isOverwritten, color) {
        if (isOverwritten === void 0) { isOverwritten = false; }
        if (color === void 0) { color = new cubismrenderer_1.CubismTextureColor(); }
        this.isOverwritten = isOverwritten;
        this.Color = color;
    }
    return DrawableColorData;
}());
exports.DrawableColorData = DrawableColorData;
/**
 * @brief テクスチャの色をRGBAで扱うための構造体
 */
var PartColorData = /** @class */ (function () {
    function PartColorData(isOverwritten, color) {
        if (isOverwritten === void 0) { isOverwritten = false; }
        if (color === void 0) { color = new cubismrenderer_1.CubismTextureColor(); }
        this.isOverwritten = isOverwritten;
        this.Color = color;
    }
    return PartColorData;
}());
exports.PartColorData = PartColorData;
/**
 * テクスチャのカリング設定を管理するための構造体
 */
var DrawableCullingData = /** @class */ (function () {
    /**
     * コンストラクタ
     *
     * @param isOverwritten
     * @param isCulling
     */
    function DrawableCullingData(isOverwritten, isCulling) {
        if (isOverwritten === void 0) { isOverwritten = false; }
        if (isCulling === void 0) { isCulling = false; }
        this.isOverwritten = isOverwritten;
        this.isCulling = isCulling;
    }
    return DrawableCullingData;
}());
exports.DrawableCullingData = DrawableCullingData;
/**
 * モデル
 *
 * Mocデータから生成されるモデルのクラス。
 */
var CubismModel = /** @class */ (function () {
    /**
     * コンストラクタ
     * @param model モデル
     */
    function CubismModel(model) {
        this._model = model;
        this._parameterValues = null;
        this._parameterMaximumValues = null;
        this._parameterMinimumValues = null;
        this._partOpacities = null;
        this._savedParameters = new csmvector_1.csmVector();
        this._parameterIds = new csmvector_1.csmVector();
        this._drawableIds = new csmvector_1.csmVector();
        this._partIds = new csmvector_1.csmVector();
        this._isOverwrittenModelMultiplyColors = false;
        this._isOverwrittenModelScreenColors = false;
        this._isOverwrittenCullings = false;
        this._modelOpacity = 1.0;
        this._userMultiplyColors = new csmvector_1.csmVector();
        this._userScreenColors = new csmvector_1.csmVector();
        this._userCullings = new csmvector_1.csmVector();
        this._userPartMultiplyColors = new csmvector_1.csmVector();
        this._userPartScreenColors = new csmvector_1.csmVector();
        this._partChildDrawables = new csmvector_1.csmVector();
        this._notExistPartId = new csmmap_1.csmMap();
        this._notExistParameterId = new csmmap_1.csmMap();
        this._notExistParameterValues = new csmmap_1.csmMap();
        this._notExistPartOpacities = new csmmap_1.csmMap();
    }
    /**
     * モデルのパラメータの更新
     */
    CubismModel.prototype.update = function () {
        // Update model
        this._model.update();
        this._model.drawables.resetDynamicFlags();
    };
    /**
     * PixelsPerUnitを取得する
     * @returns PixelsPerUnit
     */
    CubismModel.prototype.getPixelsPerUnit = function () {
        if (this._model == null) {
            return 0.0;
        }
        return this._model.canvasinfo.PixelsPerUnit;
    };
    /**
     * キャンバスの幅を取得する
     */
    CubismModel.prototype.getCanvasWidth = function () {
        if (this._model == null) {
            return 0.0;
        }
        return (this._model.canvasinfo.CanvasWidth / this._model.canvasinfo.PixelsPerUnit);
    };
    /**
     * キャンバスの高さを取得する
     */
    CubismModel.prototype.getCanvasHeight = function () {
        if (this._model == null) {
            return 0.0;
        }
        return (this._model.canvasinfo.CanvasHeight / this._model.canvasinfo.PixelsPerUnit);
    };
    /**
     * パラメータを保存する
     */
    CubismModel.prototype.saveParameters = function () {
        var parameterCount = this._model.parameters.count;
        var savedParameterCount = this._savedParameters.getSize();
        for (var i = 0; i < parameterCount; ++i) {
            if (i < savedParameterCount) {
                this._savedParameters.set(i, this._parameterValues[i]);
            }
            else {
                this._savedParameters.pushBack(this._parameterValues[i]);
            }
        }
    };
    /**
     * 乗算色を取得する
     * @param index Drawablesのインデックス
     * @returns 指定したdrawableの乗算色(RGBA)
     */
    CubismModel.prototype.getMultiplyColor = function (index) {
        // Drawableとモデル全体の乗算色上書きフラグがどちらもtrueな場合、モデル全体の上書きフラグが優先される
        if (this.getOverwriteFlagForModelMultiplyColors() ||
            this.getOverwriteFlagForDrawableMultiplyColors(index)) {
            return this._userMultiplyColors.at(index).Color;
        }
        var color = this.getDrawableMultiplyColor(index);
        return color;
    };
    /**
     * スクリーン色を取得する
     * @param index Drawablesのインデックス
     * @returns 指定したdrawableのスクリーン色(RGBA)
     */
    CubismModel.prototype.getScreenColor = function (index) {
        // Drawableとモデル全体のスクリーン色上書きフラグがどちらもtrueな場合、モデル全体の上書きフラグが優先される
        if (this.getOverwriteFlagForModelScreenColors() ||
            this.getOverwriteFlagForDrawableScreenColors(index)) {
            return this._userScreenColors.at(index).Color;
        }
        var color = this.getDrawableScreenColor(index);
        return color;
    };
    /**
     * 乗算色をセットする
     * @param index Drawablesのインデックス
     * @param color 設定する乗算色(CubismTextureColor)
     */
    CubismModel.prototype.setMultiplyColorByTextureColor = function (index, color) {
        this.setMultiplyColorByRGBA(index, color.R, color.G, color.B, color.A);
    };
    /**
     * 乗算色をセットする
     * @param index Drawablesのインデックス
     * @param r 設定する乗算色のR値
     * @param g 設定する乗算色のG値
     * @param b 設定する乗算色のB値
     * @param a 設定する乗算色のA値
     */
    CubismModel.prototype.setMultiplyColorByRGBA = function (index, r, g, b, a) {
        if (a === void 0) { a = 1.0; }
        this._userMultiplyColors.at(index).Color.R = r;
        this._userMultiplyColors.at(index).Color.G = g;
        this._userMultiplyColors.at(index).Color.B = b;
        this._userMultiplyColors.at(index).Color.A = a;
    };
    /**
     * スクリーン色をセットする
     * @param index Drawablesのインデックス
     * @param color 設定するスクリーン色(CubismTextureColor)
     */
    CubismModel.prototype.setScreenColorByTextureColor = function (index, color) {
        this.setScreenColorByRGBA(index, color.R, color.G, color.B, color.A);
    };
    /**
     * スクリーン色をセットする
     * @param index Drawablesのインデックス
     * @param r 設定するスクリーン色のR値
     * @param g 設定するスクリーン色のG値
     * @param b 設定するスクリーン色のB値
     * @param a 設定するスクリーン色のA値
     */
    CubismModel.prototype.setScreenColorByRGBA = function (index, r, g, b, a) {
        if (a === void 0) { a = 1.0; }
        this._userScreenColors.at(index).Color.R = r;
        this._userScreenColors.at(index).Color.G = g;
        this._userScreenColors.at(index).Color.B = b;
        this._userScreenColors.at(index).Color.A = a;
    };
    /**
     * partの乗算色を取得する
     * @param partIndex partのインデックス
     * @returns 指定したpartの乗算色
     */
    CubismModel.prototype.getPartMultiplyColor = function (partIndex) {
        return this._userPartMultiplyColors.at(partIndex).Color;
    };
    /**
     * partのスクリーン色を取得する
     * @param partIndex partのインデックス
     * @returns 指定したpartのスクリーン色
     */
    CubismModel.prototype.getPartScreenColor = function (partIndex) {
        return this._userPartScreenColors.at(partIndex).Color;
    };
    /**
     * partのOverwriteColor setter関数
     * @param partIndex partのインデックス
     * @param r 設定する色のR値
     * @param g 設定する色のG値
     * @param b 設定する色のB値
     * @param a 設定する色のA値
     * @param partColors 設定するpartのカラーデータ配列
     * @param drawableColors partに関連するDrawableのカラーデータ配列
     */
    CubismModel.prototype.setPartColor = function (partIndex, r, g, b, a, partColors, drawableColors) {
        partColors.at(partIndex).Color.R = r;
        partColors.at(partIndex).Color.G = g;
        partColors.at(partIndex).Color.B = b;
        partColors.at(partIndex).Color.A = a;
        if (partColors.at(partIndex).isOverwritten) {
            for (var i = 0; i < this._partChildDrawables.at(partIndex).getSize(); ++i) {
                var drawableIndex = this._partChildDrawables.at(partIndex).at(i);
                drawableColors.at(drawableIndex).Color.R = r;
                drawableColors.at(drawableIndex).Color.G = g;
                drawableColors.at(drawableIndex).Color.B = b;
                drawableColors.at(drawableIndex).Color.A = a;
            }
        }
    };
    /**
     * 乗算色をセットする
     * @param partIndex partのインデックス
     * @param color 設定する乗算色(CubismTextureColor)
     */
    CubismModel.prototype.setPartMultiplyColorByTextureColor = function (partIndex, color) {
        this.setPartMultiplyColorByRGBA(partIndex, color.R, color.G, color.B, color.A);
    };
    /**
     * 乗算色をセットする
     * @param partIndex partのインデックス
     * @param r 設定する乗算色のR値
     * @param g 設定する乗算色のG値
     * @param b 設定する乗算色のB値
     * @param a 設定する乗算色のA値
     */
    CubismModel.prototype.setPartMultiplyColorByRGBA = function (partIndex, r, g, b, a) {
        this.setPartColor(partIndex, r, g, b, a, this._userPartMultiplyColors, this._userMultiplyColors);
    };
    /**
     * スクリーン色をセットする
     * @param partIndex partのインデックス
     * @param color 設定するスクリーン色(CubismTextureColor)
     */
    CubismModel.prototype.setPartScreenColorByTextureColor = function (partIndex, color) {
        this.setPartScreenColorByRGBA(partIndex, color.R, color.G, color.B, color.A);
    };
    /**
     * スクリーン色をセットする
     * @param partIndex partのインデックス
     * @param r 設定するスクリーン色のR値
     * @param g 設定するスクリーン色のG値
     * @param b 設定するスクリーン色のB値
     * @param a 設定するスクリーン色のA値
     */
    CubismModel.prototype.setPartScreenColorByRGBA = function (partIndex, r, g, b, a) {
        this.setPartColor(partIndex, r, g, b, a, this._userPartScreenColors, this._userScreenColors);
    };
    /**
     * SDKから指定したモデルの乗算色を上書きするか
     * @returns true -> SDKからの情報を優先する
     *          false -> モデルに設定されている色情報を使用
     */
    CubismModel.prototype.getOverwriteFlagForModelMultiplyColors = function () {
        return this._isOverwrittenModelMultiplyColors;
    };
    /**
     * SDKから指定したモデルのスクリーン色を上書きするか
     * @returns true -> SDKからの情報を優先する
     *          false -> モデルに設定されている色情報を使用
     */
    CubismModel.prototype.getOverwriteFlagForModelScreenColors = function () {
        return this._isOverwrittenModelScreenColors;
    };
    /**
     * SDKから指定したモデルの乗算色を上書きするかセットする
     * @param value true -> SDKからの情報を優先する
     *              false -> モデルに設定されている色情報を使用
     */
    CubismModel.prototype.setOverwriteFlagForModelMultiplyColors = function (value) {
        this._isOverwrittenModelMultiplyColors = value;
    };
    /**
     * SDKから指定したモデルのスクリーン色を上書きするかセットする
     * @param value true -> SDKからの情報を優先する
     *              false -> モデルに設定されている色情報を使用
     */
    CubismModel.prototype.setOverwriteFlagForModelScreenColors = function (value) {
        this._isOverwrittenModelScreenColors = value;
    };
    /**
     * SDKから指定したDrawableIndexの乗算色を上書きするか
     * @returns true -> SDKからの情報を優先する
     *          false -> モデルに設定されている色情報を使用
     */
    CubismModel.prototype.getOverwriteFlagForDrawableMultiplyColors = function (drawableindex) {
        return this._userMultiplyColors.at(drawableindex).isOverwritten;
    };
    /**
     * SDKから指定したDrawableIndexのスクリーン色を上書きするか
     * @returns true -> SDKからの情報を優先する
     *          false -> モデルに設定されている色情報を使用
     */
    CubismModel.prototype.getOverwriteFlagForDrawableScreenColors = function (drawableindex) {
        return this._userScreenColors.at(drawableindex).isOverwritten;
    };
    /**
     * SDKから指定したDrawableIndexの乗算色を上書きするかセットする
     * @param value true -> SDKからの情報を優先する
     *              false -> モデルに設定されている色情報を使用
     */
    CubismModel.prototype.setOverwriteFlagForDrawableMultiplyColors = function (drawableindex, value) {
        this._userMultiplyColors.at(drawableindex).isOverwritten = value;
    };
    /**
     * SDKから指定したDrawableIndexのスクリーン色を上書きするかセットする
     * @param value true -> SDKからの情報を優先する
     *              false -> モデルに設定されている色情報を使用
     */
    CubismModel.prototype.setOverwriteFlagForDrawableScreenColors = function (drawableindex, value) {
        this._userScreenColors.at(drawableindex).isOverwritten = value;
    };
    /**
     * SDKからpartの乗算色を上書きするか
     * @param partIndex partのインデックス
     * @returns true    ->  SDKからの情報を優先する
     *          false   ->  モデルに設定されている色情報を使用
     */
    CubismModel.prototype.getOverwriteColorForPartMultiplyColors = function (partIndex) {
        return this._userPartMultiplyColors.at(partIndex).isOverwritten;
    };
    /**
     * SDKからpartのスクリーン色を上書きするか
     * @param partIndex partのインデックス
     * @returns true    ->  SDKからの情報を優先する
     *          false   ->  モデルに設定されている色情報を使用
     */
    CubismModel.prototype.getOverwriteColorForPartScreenColors = function (partIndex) {
        return this._userPartScreenColors.at(partIndex).isOverwritten;
    };
    /**
     * partのOverwriteFlag setter関数
     * @param partIndex partのインデックス
     * @param value true -> SDKからの情報を優先する
     *              false -> モデルに設定されている色情報を使用
     * @param partColors 設定するpartのカラーデータ配列
     * @param drawableColors partに関連するDrawableのカラーデータ配列
     */
    CubismModel.prototype.setOverwriteColorForPartColors = function (partIndex, value, partColors, drawableColors) {
        partColors.at(partIndex).isOverwritten = value;
        for (var i = 0; i < this._partChildDrawables.at(partIndex).getSize(); ++i) {
            var drawableIndex = this._partChildDrawables.at(partIndex).at(i);
            drawableColors.at(drawableIndex).isOverwritten = value;
            if (value) {
                drawableColors.at(drawableIndex).Color.R =
                    partColors.at(partIndex).Color.R;
                drawableColors.at(drawableIndex).Color.G =
                    partColors.at(partIndex).Color.G;
                drawableColors.at(drawableIndex).Color.B =
                    partColors.at(partIndex).Color.B;
                drawableColors.at(drawableIndex).Color.A =
                    partColors.at(partIndex).Color.A;
            }
        }
    };
    /**
     * SDKからpartのスクリーン色を上書きするかをセットする
     * @param partIndex partのインデックス
     * @param value true -> SDKからの情報を優先する
     *              false -> モデルに設定されている色情報を使用
     */
    CubismModel.prototype.setOverwriteColorForPartMultiplyColors = function (partIndex, value) {
        this._userPartMultiplyColors.at(partIndex).isOverwritten = value;
        this.setOverwriteColorForPartColors(partIndex, value, this._userPartMultiplyColors, this._userMultiplyColors);
    };
    /**
     * SDKからpartのスクリーン色を上書きするかをセットする
     * @param partIndex partのインデックス
     * @param value true -> SDKからの情報を優先する
     *              false -> モデルに設定されている色情報を使用
     */
    CubismModel.prototype.setOverwriteColorForPartScreenColors = function (partIndex, value) {
        this._userPartScreenColors.at(partIndex).isOverwritten = value;
        this.setOverwriteColorForPartColors(partIndex, value, this._userPartScreenColors, this._userScreenColors);
    };
    /**
     * Drawableのカリング情報を取得する。
     *
     * @param   drawableIndex   Drawableのインデックス
     * @return  Drawableのカリング情報
     */
    CubismModel.prototype.getDrawableCulling = function (drawableIndex) {
        if (this.getOverwriteFlagForModelCullings() ||
            this.getOverwriteFlagForDrawableCullings(drawableIndex)) {
            return this._userCullings.at(drawableIndex).isCulling;
        }
        var constantFlags = this._model.drawables.constantFlags;
        return !Live2DCubismCore.Utils.hasIsDoubleSidedBit(constantFlags[drawableIndex]);
    };
    /**
     * Drawableのカリング情報を設定する。
     *
     * @param drawableIndex Drawableのインデックス
     * @param isCulling カリング情報
     */
    CubismModel.prototype.setDrawableCulling = function (drawableIndex, isCulling) {
        this._userCullings.at(drawableIndex).isCulling = isCulling;
    };
    /**
     * SDKからモデル全体のカリング設定を上書きするか。
     *
     * @retval  true    ->  SDK上のカリング設定を使用
     * @retval  false   ->  モデルのカリング設定を使用
     */
    CubismModel.prototype.getOverwriteFlagForModelCullings = function () {
        return this._isOverwrittenCullings;
    };
    /**
     * SDKからモデル全体のカリング設定を上書きするかを設定する。
     *
     * @param isOverwrittenCullings SDK上のカリング設定を使うならtrue、モデルのカリング設定を使うならfalse
     */
    CubismModel.prototype.setOverwriteFlagForModelCullings = function (isOverwrittenCullings) {
        this._isOverwrittenCullings = isOverwrittenCullings;
    };
    /**
     *
     * @param drawableIndex Drawableのインデックス
     * @retval  true    ->  SDK上のカリング設定を使用
     * @retval  false   ->  モデルのカリング設定を使用
     */
    CubismModel.prototype.getOverwriteFlagForDrawableCullings = function (drawableIndex) {
        return this._userCullings.at(drawableIndex).isOverwritten;
    };
    /**
     *
     * @param drawableIndex Drawableのインデックス
     * @param isOverwrittenCullings SDK上のカリング設定を使うならtrue、モデルのカリング設定を使うならfalse
     */
    CubismModel.prototype.setOverwriteFlagForDrawableCullings = function (drawableIndex, isOverwrittenCullings) {
        this._userCullings.at(drawableIndex).isOverwritten = isOverwrittenCullings;
    };
    /**
     * モデルの不透明度を取得する
     *
     * @returns 不透明度の値
     */
    CubismModel.prototype.getModelOapcity = function () {
        return this._modelOpacity;
    };
    /**
     * モデルの不透明度を設定する
     *
     * @param value 不透明度の値
     */
    CubismModel.prototype.setModelOapcity = function (value) {
        this._modelOpacity = value;
    };
    /**
     * モデルを取得
     */
    CubismModel.prototype.getModel = function () {
        return this._model;
    };
    /**
     * パーツのインデックスを取得
     * @param partId パーツのID
     * @return パーツのインデックス
     */
    CubismModel.prototype.getPartIndex = function (partId) {
        var partIndex;
        var partCount = this._model.parts.count;
        for (partIndex = 0; partIndex < partCount; ++partIndex) {
            if (partId == this._partIds.at(partIndex)) {
                return partIndex;
            }
        }
        // モデルに存在していない場合、非存在パーツIDリスト内にあるかを検索し、そのインデックスを返す
        if (this._notExistPartId.isExist(partId)) {
            return this._notExistPartId.getValue(partId);
        }
        // 非存在パーツIDリストにない場合、新しく要素を追加する
        partIndex = partCount + this._notExistPartId.getSize();
        this._notExistPartId.setValue(partId, partIndex);
        this._notExistPartOpacities.appendKey(partIndex);
        return partIndex;
    };
    /**
     * パーツのIDを取得する。
     *
     * @param partIndex 取得するパーツのインデックス
     * @return パーツのID
     */
    CubismModel.prototype.getPartId = function (partIndex) {
        var partId = this._model.parts.ids[partIndex];
        return live2dcubismframework_1.CubismFramework.getIdManager().getId(partId);
    };
    /**
     * パーツの個数の取得
     * @return パーツの個数
     */
    CubismModel.prototype.getPartCount = function () {
        var partCount = this._model.parts.count;
        return partCount;
    };
    /**
     * パーツの不透明度の設定(Index)
     * @param partIndex パーツのインデックス
     * @param opacity 不透明度
     */
    CubismModel.prototype.setPartOpacityByIndex = function (partIndex, opacity) {
        if (this._notExistPartOpacities.isExist(partIndex)) {
            this._notExistPartOpacities.setValue(partIndex, opacity);
            return;
        }
        // インデックスの範囲内検知
        (0, cubismdebug_1.CSM_ASSERT)(0 <= partIndex && partIndex < this.getPartCount());
        this._partOpacities[partIndex] = opacity;
    };
    /**
     * パーツの不透明度の設定(Id)
     * @param partId パーツのID
     * @param opacity パーツの不透明度
     */
    CubismModel.prototype.setPartOpacityById = function (partId, opacity) {
        // 高速化のためにPartIndexを取得できる機構になっているが、外部からの設定の時は呼び出し頻度が低いため不要
        var index = this.getPartIndex(partId);
        if (index < 0) {
            return; // パーツがないのでスキップ
        }
        this.setPartOpacityByIndex(index, opacity);
    };
    /**
     * パーツの不透明度の取得(index)
     * @param partIndex パーツのインデックス
     * @return パーツの不透明度
     */
    CubismModel.prototype.getPartOpacityByIndex = function (partIndex) {
        if (this._notExistPartOpacities.isExist(partIndex)) {
            // モデルに存在しないパーツIDの場合、非存在パーツリストから不透明度を返す。
            return this._notExistPartOpacities.getValue(partIndex);
        }
        // インデックスの範囲内検知
        (0, cubismdebug_1.CSM_ASSERT)(0 <= partIndex && partIndex < this.getPartCount());
        return this._partOpacities[partIndex];
    };
    /**
     * パーツの不透明度の取得(id)
     * @param partId パーツのＩｄ
     * @return パーツの不透明度
     */
    CubismModel.prototype.getPartOpacityById = function (partId) {
        // 高速化のためにPartIndexを取得できる機構になっているが、外部からの設定の時は呼び出し頻度が低いため不要
        var index = this.getPartIndex(partId);
        if (index < 0) {
            return 0; // パーツが無いのでスキップ
        }
        return this.getPartOpacityByIndex(index);
    };
    /**
     * パラメータのインデックスの取得
     * @param パラメータID
     * @return パラメータのインデックス
     */
    CubismModel.prototype.getParameterIndex = function (parameterId) {
        var parameterIndex;
        var idCount = this._model.parameters.count;
        for (parameterIndex = 0; parameterIndex < idCount; ++parameterIndex) {
            if (parameterId != this._parameterIds.at(parameterIndex)) {
                continue;
            }
            return parameterIndex;
        }
        // モデルに存在していない場合、非存在パラメータIDリスト内を検索し、そのインデックスを返す
        if (this._notExistParameterId.isExist(parameterId)) {
            return this._notExistParameterId.getValue(parameterId);
        }
        // 非存在パラメータIDリストにない場合新しく要素を追加する
        parameterIndex =
            this._model.parameters.count + this._notExistParameterId.getSize();
        this._notExistParameterId.setValue(parameterId, parameterIndex);
        this._notExistParameterValues.appendKey(parameterIndex);
        return parameterIndex;
    };
    /**
     * パラメータの個数の取得
     * @return パラメータの個数
     */
    CubismModel.prototype.getParameterCount = function () {
        return this._model.parameters.count;
    };
    /**
     * パラメータの種類の取得
     * @param parameterIndex パラメータのインデックス
     * @return csmParameterType_Normal -> 通常のパラメータ
     *          csmParameterType_BlendShape -> ブレンドシェイプパラメータ
     */
    CubismModel.prototype.getParameterType = function (parameterIndex) {
        return this._model.parameters.types[parameterIndex];
    };
    /**
     * パラメータの最大値の取得
     * @param parameterIndex パラメータのインデックス
     * @return パラメータの最大値
     */
    CubismModel.prototype.getParameterMaximumValue = function (parameterIndex) {
        return this._model.parameters.maximumValues[parameterIndex];
    };
    /**
     * パラメータの最小値の取得
     * @param parameterIndex パラメータのインデックス
     * @return パラメータの最小値
     */
    CubismModel.prototype.getParameterMinimumValue = function (parameterIndex) {
        return this._model.parameters.minimumValues[parameterIndex];
    };
    /**
     * パラメータのデフォルト値の取得
     * @param parameterIndex パラメータのインデックス
     * @return パラメータのデフォルト値
     */
    CubismModel.prototype.getParameterDefaultValue = function (parameterIndex) {
        return this._model.parameters.defaultValues[parameterIndex];
    };
    /**
     * パラメータの値の取得
     * @param parameterIndex    パラメータのインデックス
     * @return パラメータの値
     */
    CubismModel.prototype.getParameterValueByIndex = function (parameterIndex) {
        if (this._notExistParameterValues.isExist(parameterIndex)) {
            return this._notExistParameterValues.getValue(parameterIndex);
        }
        // インデックスの範囲内検知
        (0, cubismdebug_1.CSM_ASSERT)(0 <= parameterIndex && parameterIndex < this.getParameterCount());
        return this._parameterValues[parameterIndex];
    };
    /**
     * パラメータの値の取得
     * @param parameterId    パラメータのID
     * @return パラメータの値
     */
    CubismModel.prototype.getParameterValueById = function (parameterId) {
        // 高速化のためにparameterIndexを取得できる機構になっているが、外部からの設定の時は呼び出し頻度が低いため不要
        var parameterIndex = this.getParameterIndex(parameterId);
        return this.getParameterValueByIndex(parameterIndex);
    };
    /**
     * パラメータの値の設定
     * @param parameterIndex パラメータのインデックス
     * @param value パラメータの値
     * @param weight 重み
     */
    CubismModel.prototype.setParameterValueByIndex = function (parameterIndex, value, weight) {
        if (weight === void 0) { weight = 1.0; }
        if (this._notExistParameterValues.isExist(parameterIndex)) {
            this._notExistParameterValues.setValue(parameterIndex, weight == 1
                ? value
                : this._notExistParameterValues.getValue(parameterIndex) *
                    (1 - weight) +
                    value * weight);
            return;
        }
        // インデックスの範囲内検知
        (0, cubismdebug_1.CSM_ASSERT)(0 <= parameterIndex && parameterIndex < this.getParameterCount());
        if (this._model.parameters.maximumValues[parameterIndex] < value) {
            value = this._model.parameters.maximumValues[parameterIndex];
        }
        if (this._model.parameters.minimumValues[parameterIndex] > value) {
            value = this._model.parameters.minimumValues[parameterIndex];
        }
        this._parameterValues[parameterIndex] =
            weight == 1
                ? value
                : (this._parameterValues[parameterIndex] =
                    this._parameterValues[parameterIndex] * (1 - weight) +
                        value * weight);
    };
    /**
     * パラメータの値の設定
     * @param parameterId パラメータのID
     * @param value パラメータの値
     * @param weight 重み
     */
    CubismModel.prototype.setParameterValueById = function (parameterId, value, weight) {
        if (weight === void 0) { weight = 1.0; }
        var index = this.getParameterIndex(parameterId);
        this.setParameterValueByIndex(index, value, weight);
    };
    /**
     * パラメータの値の加算(index)
     * @param parameterIndex パラメータインデックス
     * @param value 加算する値
     * @param weight 重み
     */
    CubismModel.prototype.addParameterValueByIndex = function (parameterIndex, value, weight) {
        if (weight === void 0) { weight = 1.0; }
        this.setParameterValueByIndex(parameterIndex, this.getParameterValueByIndex(parameterIndex) + value * weight);
    };
    /**
     * パラメータの値の加算(id)
     * @param parameterId パラメータＩＤ
     * @param value 加算する値
     * @param weight 重み
     */
    CubismModel.prototype.addParameterValueById = function (parameterId, value, weight) {
        if (weight === void 0) { weight = 1.0; }
        var index = this.getParameterIndex(parameterId);
        this.addParameterValueByIndex(index, value, weight);
    };
    /**
     * パラメータの値の乗算
     * @param parameterId パラメータのID
     * @param value 乗算する値
     * @param weight 重み
     */
    CubismModel.prototype.multiplyParameterValueById = function (parameterId, value, weight) {
        if (weight === void 0) { weight = 1.0; }
        var index = this.getParameterIndex(parameterId);
        this.multiplyParameterValueByIndex(index, value, weight);
    };
    /**
     * パラメータの値の乗算
     * @param parameterIndex パラメータのインデックス
     * @param value 乗算する値
     * @param weight 重み
     */
    CubismModel.prototype.multiplyParameterValueByIndex = function (parameterIndex, value, weight) {
        if (weight === void 0) { weight = 1.0; }
        this.setParameterValueByIndex(parameterIndex, this.getParameterValueByIndex(parameterIndex) *
            (1.0 + (value - 1.0) * weight));
    };
    /**
     * Drawableのインデックスの取得
     * @param drawableId DrawableのID
     * @return Drawableのインデックス
     */
    CubismModel.prototype.getDrawableIndex = function (drawableId) {
        var drawableCount = this._model.drawables.count;
        for (var drawableIndex = 0; drawableIndex < drawableCount; ++drawableIndex) {
            if (this._drawableIds.at(drawableIndex) == drawableId) {
                return drawableIndex;
            }
        }
        return -1;
    };
    /**
     * Drawableの個数の取得
     * @return drawableの個数
     */
    CubismModel.prototype.getDrawableCount = function () {
        var drawableCount = this._model.drawables.count;
        return drawableCount;
    };
    /**
     * DrawableのIDを取得する
     * @param drawableIndex Drawableのインデックス
     * @return drawableのID
     */
    CubismModel.prototype.getDrawableId = function (drawableIndex) {
        var parameterIds = this._model.drawables.ids;
        return live2dcubismframework_1.CubismFramework.getIdManager().getId(parameterIds[drawableIndex]);
    };
    /**
     * Drawableの描画順リストの取得
     * @return Drawableの描画順リスト
     */
    CubismModel.prototype.getDrawableRenderOrders = function () {
        var renderOrders = this._model.drawables.renderOrders;
        return renderOrders;
    };
    /**
     * @deprecated
     * 関数名が誤っていたため、代替となる getDrawableTextureIndex を追加し、この関数は非推奨となりました。
     *
     * Drawableのテクスチャインデックスリストの取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableのテクスチャインデックスリスト
     */
    CubismModel.prototype.getDrawableTextureIndices = function (drawableIndex) {
        return this.getDrawableTextureIndex(drawableIndex);
    };
    /**
     * Drawableのテクスチャインデックスの取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableのテクスチャインデックス
     */
    CubismModel.prototype.getDrawableTextureIndex = function (drawableIndex) {
        var textureIndices = this._model.drawables.textureIndices;
        return textureIndices[drawableIndex];
    };
    /**
     * DrawableのVertexPositionsの変化情報の取得
     *
     * 直近のCubismModel.update関数でDrawableの頂点情報が変化したかを取得する。
     *
     * @param   drawableIndex   Drawableのインデックス
     * @retval  true    Drawableの頂点情報が直近のCubismModel.update関数で変化した
     * @retval  false   Drawableの頂点情報が直近のCubismModel.update関数で変化していない
     */
    CubismModel.prototype.getDrawableDynamicFlagVertexPositionsDidChange = function (drawableIndex) {
        var dynamicFlags = this._model.drawables.dynamicFlags;
        return Live2DCubismCore.Utils.hasVertexPositionsDidChangeBit(dynamicFlags[drawableIndex]);
    };
    /**
     * Drawableの頂点インデックスの個数の取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableの頂点インデックスの個数
     */
    CubismModel.prototype.getDrawableVertexIndexCount = function (drawableIndex) {
        var indexCounts = this._model.drawables.indexCounts;
        return indexCounts[drawableIndex];
    };
    /**
     * Drawableの頂点の個数の取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableの頂点の個数
     */
    CubismModel.prototype.getDrawableVertexCount = function (drawableIndex) {
        var vertexCounts = this._model.drawables.vertexCounts;
        return vertexCounts[drawableIndex];
    };
    /**
     * Drawableの頂点リストの取得
     * @param drawableIndex drawableのインデックス
     * @return drawableの頂点リスト
     */
    CubismModel.prototype.getDrawableVertices = function (drawableIndex) {
        return this.getDrawableVertexPositions(drawableIndex);
    };
    /**
     * Drawableの頂点インデックスリストの取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableの頂点インデックスリスト
     */
    CubismModel.prototype.getDrawableVertexIndices = function (drawableIndex) {
        var indicesArray = this._model.drawables.indices;
        return indicesArray[drawableIndex];
    };
    /**
     * Drawableの頂点リストの取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableの頂点リスト
     */
    CubismModel.prototype.getDrawableVertexPositions = function (drawableIndex) {
        var verticesArray = this._model.drawables.vertexPositions;
        return verticesArray[drawableIndex];
    };
    /**
     * Drawableの頂点のUVリストの取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableの頂点UVリスト
     */
    CubismModel.prototype.getDrawableVertexUvs = function (drawableIndex) {
        var uvsArray = this._model.drawables.vertexUvs;
        return uvsArray[drawableIndex];
    };
    /**
     * Drawableの不透明度の取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableの不透明度
     */
    CubismModel.prototype.getDrawableOpacity = function (drawableIndex) {
        var opacities = this._model.drawables.opacities;
        return opacities[drawableIndex];
    };
    /**
     * Drawableの乗算色の取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableの乗算色(RGBA)
     * スクリーン色はRGBAで取得されるが、Aは必ず0
     */
    CubismModel.prototype.getDrawableMultiplyColor = function (drawableIndex) {
        var multiplyColors = this._model.drawables.multiplyColors;
        var index = drawableIndex * 4;
        var multiplyColor = new cubismrenderer_1.CubismTextureColor();
        multiplyColor.R = multiplyColors[index];
        multiplyColor.G = multiplyColors[index + 1];
        multiplyColor.B = multiplyColors[index + 2];
        multiplyColor.A = multiplyColors[index + 3];
        return multiplyColor;
    };
    /**
     * Drawableのスクリーン色の取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableのスクリーン色(RGBA)
     * スクリーン色はRGBAで取得されるが、Aは必ず0
     */
    CubismModel.prototype.getDrawableScreenColor = function (drawableIndex) {
        var screenColors = this._model.drawables.screenColors;
        var index = drawableIndex * 4;
        var screenColor = new cubismrenderer_1.CubismTextureColor();
        screenColor.R = screenColors[index];
        screenColor.G = screenColors[index + 1];
        screenColor.B = screenColors[index + 2];
        screenColor.A = screenColors[index + 3];
        return screenColor;
    };
    /**
     * Drawableの親パーツのインデックスの取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableの親パーツのインデックス
     */
    CubismModel.prototype.getDrawableParentPartIndex = function (drawableIndex) {
        return this._model.drawables.parentPartIndices[drawableIndex];
    };
    /**
     * Drawableのブレンドモードを取得
     * @param drawableIndex Drawableのインデックス
     * @return drawableのブレンドモード
     */
    CubismModel.prototype.getDrawableBlendMode = function (drawableIndex) {
        var constantFlags = this._model.drawables.constantFlags;
        return Live2DCubismCore.Utils.hasBlendAdditiveBit(constantFlags[drawableIndex])
            ? cubismrenderer_1.CubismBlendMode.CubismBlendMode_Additive
            : Live2DCubismCore.Utils.hasBlendMultiplicativeBit(constantFlags[drawableIndex])
                ? cubismrenderer_1.CubismBlendMode.CubismBlendMode_Multiplicative
                : cubismrenderer_1.CubismBlendMode.CubismBlendMode_Normal;
    };
    /**
     * Drawableのマスクの反転使用の取得
     *
     * Drawableのマスク使用時の反転設定を取得する。
     * マスクを使用しない場合は無視される。
     *
     * @param drawableIndex Drawableのインデックス
     * @return Drawableの反転設定
     */
    CubismModel.prototype.getDrawableInvertedMaskBit = function (drawableIndex) {
        var constantFlags = this._model.drawables.constantFlags;
        return Live2DCubismCore.Utils.hasIsInvertedMaskBit(constantFlags[drawableIndex]);
    };
    /**
     * Drawableのクリッピングマスクリストの取得
     * @return Drawableのクリッピングマスクリスト
     */
    CubismModel.prototype.getDrawableMasks = function () {
        var masks = this._model.drawables.masks;
        return masks;
    };
    /**
     * Drawableのクリッピングマスクの個数リストの取得
     * @return Drawableのクリッピングマスクの個数リスト
     */
    CubismModel.prototype.getDrawableMaskCounts = function () {
        var maskCounts = this._model.drawables.maskCounts;
        return maskCounts;
    };
    /**
     * クリッピングマスクの使用状態
     *
     * @return true クリッピングマスクを使用している
     * @return false クリッピングマスクを使用していない
     */
    CubismModel.prototype.isUsingMasking = function () {
        for (var d = 0; d < this._model.drawables.count; ++d) {
            if (this._model.drawables.maskCounts[d] <= 0) {
                continue;
            }
            return true;
        }
        return false;
    };
    /**
     * Drawableの表示情報を取得する
     *
     * @param drawableIndex Drawableのインデックス
     * @return true Drawableが表示
     * @return false Drawableが非表示
     */
    CubismModel.prototype.getDrawableDynamicFlagIsVisible = function (drawableIndex) {
        var dynamicFlags = this._model.drawables.dynamicFlags;
        return Live2DCubismCore.Utils.hasIsVisibleBit(dynamicFlags[drawableIndex]);
    };
    /**
     * DrawableのDrawOrderの変化情報の取得
     *
     * 直近のCubismModel.update関数でdrawableのdrawOrderが変化したかを取得する。
     * drawOrderはartMesh上で指定する0から1000の情報
     * @param drawableIndex drawableのインデックス
     * @return true drawableの不透明度が直近のCubismModel.update関数で変化した
     * @return false drawableの不透明度が直近のCubismModel.update関数で変化している
     */
    CubismModel.prototype.getDrawableDynamicFlagVisibilityDidChange = function (drawableIndex) {
        var dynamicFlags = this._model.drawables.dynamicFlags;
        return Live2DCubismCore.Utils.hasVisibilityDidChangeBit(dynamicFlags[drawableIndex]);
    };
    /**
     * Drawableの不透明度の変化情報の取得
     *
     * 直近のCubismModel.update関数でdrawableの不透明度が変化したかを取得する。
     *
     * @param drawableIndex drawableのインデックス
     * @return true Drawableの不透明度が直近のCubismModel.update関数で変化した
     * @return false Drawableの不透明度が直近のCubismModel.update関数で変化してない
     */
    CubismModel.prototype.getDrawableDynamicFlagOpacityDidChange = function (drawableIndex) {
        var dynamicFlags = this._model.drawables.dynamicFlags;
        return Live2DCubismCore.Utils.hasOpacityDidChangeBit(dynamicFlags[drawableIndex]);
    };
    /**
     * Drawableの描画順序の変化情報の取得
     *
     * 直近のCubismModel.update関数でDrawableの描画の順序が変化したかを取得する。
     *
     * @param drawableIndex Drawableのインデックス
     * @return true Drawableの描画の順序が直近のCubismModel.update関数で変化した
     * @return false Drawableの描画の順序が直近のCubismModel.update関数で変化してない
     */
    CubismModel.prototype.getDrawableDynamicFlagRenderOrderDidChange = function (drawableIndex) {
        var dynamicFlags = this._model.drawables.dynamicFlags;
        return Live2DCubismCore.Utils.hasRenderOrderDidChangeBit(dynamicFlags[drawableIndex]);
    };
    /**
     * Drawableの乗算色・スクリーン色の変化情報の取得
     *
     * 直近のCubismModel.update関数でDrawableの乗算色・スクリーン色が変化したかを取得する。
     *
     * @param drawableIndex Drawableのインデックス
     * @return true Drawableの乗算色・スクリーン色が直近のCubismModel.update関数で変化した
     * @return false Drawableの乗算色・スクリーン色が直近のCubismModel.update関数で変化してない
     */
    CubismModel.prototype.getDrawableDynamicFlagBlendColorDidChange = function (drawableIndex) {
        var dynamicFlags = this._model.drawables.dynamicFlags;
        return Live2DCubismCore.Utils.hasBlendColorDidChangeBit(dynamicFlags[drawableIndex]);
    };
    /**
     * 保存されたパラメータの読み込み
     */
    CubismModel.prototype.loadParameters = function () {
        var parameterCount = this._model.parameters.count;
        var savedParameterCount = this._savedParameters.getSize();
        if (parameterCount > savedParameterCount) {
            parameterCount = savedParameterCount;
        }
        for (var i = 0; i < parameterCount; ++i) {
            this._parameterValues[i] = this._savedParameters.at(i);
        }
    };
    /**
     * 初期化する
     */
    CubismModel.prototype.initialize = function () {
        (0, cubismdebug_1.CSM_ASSERT)(this._model);
        this._parameterValues = this._model.parameters.values;
        this._partOpacities = this._model.parts.opacities;
        this._parameterMaximumValues = this._model.parameters.maximumValues;
        this._parameterMinimumValues = this._model.parameters.minimumValues;
        {
            var parameterIds = this._model.parameters.ids;
            var parameterCount = this._model.parameters.count;
            this._parameterIds.prepareCapacity(parameterCount);
            for (var i = 0; i < parameterCount; ++i) {
                this._parameterIds.pushBack(live2dcubismframework_1.CubismFramework.getIdManager().getId(parameterIds[i]));
            }
        }
        var partCount = this._model.parts.count;
        {
            var partIds = this._model.parts.ids;
            this._partIds.prepareCapacity(partCount);
            for (var i = 0; i < partCount; ++i) {
                this._partIds.pushBack(live2dcubismframework_1.CubismFramework.getIdManager().getId(partIds[i]));
            }
            this._userPartMultiplyColors.prepareCapacity(partCount);
            this._userPartScreenColors.prepareCapacity(partCount);
            this._partChildDrawables.prepareCapacity(partCount);
        }
        {
            var drawableIds = this._model.drawables.ids;
            var drawableCount = this._model.drawables.count;
            this._userMultiplyColors.prepareCapacity(drawableCount);
            this._userScreenColors.prepareCapacity(drawableCount);
            // カリング設定
            this._userCullings.prepareCapacity(drawableCount);
            var userCulling = new DrawableCullingData(false, false);
            // Part
            {
                for (var i = 0; i < partCount; ++i) {
                    var multiplyColor = new cubismrenderer_1.CubismTextureColor(1.0, 1.0, 1.0, 1.0);
                    var screenColor = new cubismrenderer_1.CubismTextureColor(0.0, 0.0, 0.0, 1.0);
                    var userMultiplyColor = new PartColorData(false, multiplyColor);
                    var userScreenColor = new PartColorData(false, screenColor);
                    this._userPartMultiplyColors.pushBack(userMultiplyColor);
                    this._userPartScreenColors.pushBack(userScreenColor);
                    this._partChildDrawables.pushBack(new csmvector_1.csmVector());
                    this._partChildDrawables.at(i).prepareCapacity(drawableCount);
                }
            }
            // Drawables
            {
                for (var i = 0; i < drawableCount; ++i) {
                    var multiplyColor = new cubismrenderer_1.CubismTextureColor(1.0, 1.0, 1.0, 1.0);
                    var screenColor = new cubismrenderer_1.CubismTextureColor(0.0, 0.0, 0.0, 1.0);
                    var userMultiplyColor = new DrawableColorData(false, multiplyColor);
                    var userScreenColor = new DrawableColorData(false, screenColor);
                    this._drawableIds.pushBack(live2dcubismframework_1.CubismFramework.getIdManager().getId(drawableIds[i]));
                    this._userMultiplyColors.pushBack(userMultiplyColor);
                    this._userScreenColors.pushBack(userScreenColor);
                    this._userCullings.pushBack(userCulling);
                    var parentIndex = this.getDrawableParentPartIndex(i);
                    if (parentIndex >= 0) {
                        this._partChildDrawables.at(parentIndex).pushBack(i);
                    }
                }
            }
        }
    };
    /**
     * デストラクタ相当の処理
     */
    CubismModel.prototype.release = function () {
        this._model.release();
        this._model = null;
    };
    return CubismModel;
}());
exports.CubismModel = CubismModel;
// Namespace definition for compatibility.
var $ = __importStar(require("./cubismmodel"));
// eslint-disable-next-line @typescript-eslint/no-namespace
var Live2DCubismFramework;
(function (Live2DCubismFramework) {
    Live2DCubismFramework.CubismModel = $.CubismModel;
})(Live2DCubismFramework = exports.Live2DCubismFramework || (exports.Live2DCubismFramework = {}));
//# sourceMappingURL=cubismmodel.js.map