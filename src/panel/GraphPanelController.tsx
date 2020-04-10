import React from 'react';
import moment from 'moment'

import * as tf from '@tensorflow/tfjs'

import { GraphSeriesToggler, Button, Tooltip } from '@grafana/ui';
import { PanelData, GraphSeriesXY, AbsoluteTimeRange, TimeZone, AppEvents } from '@grafana/data';
import { getDataSourceSrv, getBackendSrv } from '@grafana/runtime';
import appEvents from 'grafana/app/core/app_events';

import { getGraphSeriesModel } from './getGraphSeriesModel';
import { Options, SeriesOptions } from './types';
import { SeriesColorChangeHandler, SeriesAxisToggleHandler } from '@grafana/ui/src/components/Graph/GraphWithLegend';

import {
  extract_tooltip_feature,
  extract_group_by,
  extract_fill_value,
  extract_format_tags,
  extract_is_valid,
  extract_model_database,
  extract_model_measurement,
  extract_model_select,
  extract_model_feature,
  extract_model_func,
  extract_model_fill,
  extract_model_time_format,
  extract_model_time,
  extract_model_tags,
  extract_model_tags_map
} from './extractors';

// tf.enableDebugMode();

// -- ML code ----------------------------------------------------------------
function _convertToTensor(data) {
  appEvents.emit(AppEvents.alertSuccess, ['Converting series data to Tensor']);

  return tf.tidy(() => {
    tf.util.shuffle(data);

    const inputs = data.map(d => d.y)
    const labels = data.map(d => d.x);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

function _onBatchEnd(batch, logs) {
    console.log('Accuracy', logs);
    // appEvents.emit(AppEvents.alertSuccess, ['Training batch complete']);
}

async function _fitModel(model, inputs, labels) {
  const LEARNING_RATE = 0.0001;
  const optimizer = tf.train.adam(LEARNING_RATE);
  model.compile({
    optimizer: optimizer,
    loss: tf.losses.meanSquaredError, // loss: 'categoricalCrossentropy',
    metrics: ['mse'],
  });

  console.log('Model compiled', model);

  const batchSize = 64;
  const epochs = 10;

  appEvents.emit(AppEvents.alertSuccess, ['Training model...']);

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    validationSplit: 0.1,
    callbacks: {_onBatchEnd}
  });
}

function _trainModel(model: any, source: any) {
  const timeData = source.fields[0].values.buffer;
  const metricData = source.fields[1].values.buffer;

  var data = timeData.map(function(timestamp, i) {
    return {x: timestamp, y: metricData[i]};
  });

  console.log(data);

  const tensorData = _convertToTensor(data);
  let {inputs, labels} = tensorData;

  // console.log(inputs.arraySync());

  _fitModel(model, inputs, labels).then(result => {
    console.log('Done Training', result);
    appEvents.emit(AppEvents.alertSuccess, ['Training complete']);
  });

  return tensorData;
}

function _forecastModel(model: any) {
  console.log(model);
  const { vae, tensorData } = model;
  console.log('VAE', vae);
  console.log('Tensor data', tensorData);

  const {inputMax, inputMin, labelMin, labelMax} = tensorData;

  const [preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const preds = vae.predict(xs.reshape([100, 1]));

    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormPreds.dataSync()];
  });


  const predictedPoints = Array.from(preds).map((val, i) => {
    return {y: val}
  });

  return predictedPoints;
}

function _createAndTrainModel(source: any) {
  let input_shape = [1];
  // let base_depth = [1];
  // let encoded_size = 16;

  // // VAE
  // let encoder = tf.sequential({
  //   name: 'vae_encoder',
  //   layers: [
  //     tf.layers.inputLayer({
  //       inputShape: input_shape
  //     }),
  //     // tf.layers.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
  //     tf.layers.conv2d({
  //       inputShape: base_depth,
  //       filters: 5,
  //       kernelSize: 1,
  //       strides: 1,
  //       padding: 'same',
  //       activation: 'LeakyReLU',
  //       kernelInitializer: 'varianceScaling'
  //     }),
  //     tf.layers.conv2d({
  //       inputShape: base_depth,
  //       filters: 5,
  //       kernelSize: 1,
  //       strides: 2,
  //       padding: 'same',
  //       activation: 'LeakyReLU',
  //       kernelInitializer: 'varianceScaling'
  //     }),
  //     tf.layers.conv2d({
  //       inputShape: 2 * base_depth,
  //       filters: 5,
  //       kernelSize: 1,
  //       strides: 1,
  //       padding: 'same',
  //       activation: 'LeakyReLU',
  //       kernelInitializer: 'varianceScaling'
  //     }),
  //     tf.layers.conv2d({
  //       inputShape: 2 * base_depth,
  //       filters: 5,
  //       kernelSize: 1,
  //       strides: 2,
  //       padding: 'same',
  //       activation: 'LeakyReLU',
  //       kernelInitializer: 'varianceScaling'
  //     }),
  //     tf.layers.conv2d({
  //       inputShape: 4* encoded_size,
  //       filters: 7,
  //       kernelSize: 1,
  //       strides: 1,
  //       padding: 'valid',
  //       activation: 'LeakyReLU',
  //       kernelInitializer: 'varianceScaling'
  //     }),
  //     tf.layers.flatten(),
  //     tf.layers.dense({
  //       inputShape: input_shape,
  //       units: 1,
  //       useBias: true
  //     })
  //     // tf.layers.dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
  //     //            activation=None),
  //     // tf.layers.MultivariateNormalTriL(
  //     //     encoded_size,
  //     //     activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=1.0)),
  //   ]
  // });

  // let decoder = tf.sequential({
  //   name: 'vae_decoder',
  //   layers: [
  //     tf.layers.inputLayer({
  //       inputShape: [encoded_size]
  //     }),
  //     tf.layers.reshape({
  //       targetShape: [1, 1, encoded_size]
  //     }),
  //     tf.layers.conv2dTranspose({
  //       inputShape: 2 * base_depth,
  //       filters: 7,
  //       kernelSize: 1,
  //       strides: 1,
  //       padding: 'valid',
  //       activation: 'LeakyReLU',
  //       kernelInitializer: 'varianceScaling'
  //     }),
  //     tf.layers.conv2dTranspose({
  //       inputShape: 2 * base_depth,
  //       filters: 5,
  //       kernelSize: 1,
  //       strides: 1,
  //       padding: 'same',
  //       activation: 'LeakyReLU',
  //       kernelInitializer: 'varianceScaling'
  //     }),
  //     tf.layers.conv2dTranspose({
  //       inputShape: 2 * base_depth,
  //       filters: 5,
  //       kernelSize: 1,
  //       strides: 2,
  //       padding: 'same',
  //       activation: 'LeakyReLU',
  //       kernelInitializer: 'varianceScaling'
  //     }),
  //     tf.layers.conv2dTranspose({
  //       inputShape: base_depth,
  //       filters: 5,
  //       kernelSize: 1,
  //       strides: 1,
  //       padding: 'same',
  //       activation: 'LeakyReLU',
  //       kernelInitializer: 'varianceScaling'
  //     }),
  //     tf.layers.conv2dTranspose({
  //       inputShape: base_depth,
  //       filters: 5,
  //       kernelSize: 1,
  //       strides: 2,
  //       padding: 'same',
  //       activation: 'LeakyReLU',
  //       kernelInitializer: 'varianceScaling'
  //     }),
  //     tf.layers.conv2dTranspose({
  //       inputShape: base_depth,
  //       filters: 5,
  //       kernelSize: 1,
  //       strides: 1,
  //       padding: 'same',
  //       activation: 'LeakyReLU',
  //       kernelInitializer: 'varianceScaling'
  //     }),
  //     tf.layers.conv2d({
  //       filters: 1, kernelSize: 5, strides: 1, padding: 'same'
  //     }),
  //     tf.layers.flatten(),
  //     // tf.layers.IndependentBernoulli(input_shape, tf.layers.Bernoulli.logits),
  //   ]
  // });

  let encoder = tf.sequential();
  encoder.add(tf.layers.inputLayer({inputShape: input_shape}));
  encoder.add(tf.layers.dense({units: 1, useBias: true, activation: 'relu'}));
  encoder.add(tf.layers.dense({units: 1, useBias: true, activation: 'relu'}));
  encoder.add(tf.layers.dense({units: 1, useBias: true}));
  console.log('Encoder', encoder);

  let decoder = tf.sequential();
  decoder.add(tf.layers.inputLayer({inputShape: input_shape}));
  decoder.add(tf.layers.dense({units: 1, useBias: true, activation: 'relu'}));
  decoder.add(tf.layers.dense({units: 1, useBias: true, activation: 'relu'}));
  decoder.add(tf.layers.dense({units: 1, useBias: true, activation: 'linear'}));
  console.log('Decoder', decoder);

  // let encoder = tf.sequential({
  //   name: 'vae_encoder',
  //   layers: [
  //     tf.layers.inputLayer({
  //       inputShape: input_shape
  //     }),
  //     tf.layers.dense({
  //       units: 1,
  //       // kernelInitializer: tf.regularizers.l2(), // {l2: 0.001}
  //       useBias: true,
  //       activation: 'relu'
  //     }),
  //     tf.layers.dense({
  //       units: 1,
  //       // kernelInitializer: tf.regularizers.l2(), // {l2: 0.001}
  //       useBias: true,
  //       activation: 'relu'
  //     }),
  //     tf.layers.dense({
  //       units: 1,
  //       useBias: true
  //     })
  //   ]
  // });

  // let decoder = tf.sequential({
  //   name: 'vae_decoder',
  //   layers: [
  //     tf.layers.inputLayer({
  //       inputShape: input_shape
  //     }),
  //     tf.layers.dense({
  //       units: 1,
  //       // kernelInitializer: tf.regularizers.l2(), // {l2: 0.001}
  //       useBias: true,
  //       activation: 'relu'
  //     }),
  //     tf.layers.dense({
  //       units: 1,
  //       // kernelInitializer: tf.regularizers.l2(), // {l2: 0.001}
  //       useBias: true,
  //       activation: 'relu'
  //     }),
  //     tf.layers.dense({
  //       units: 1,
  //       useBias: true,
  //       activation: 'linear'
  //     })
  //   ]
  // });

  const vae = tf.sequential();
  vae.add(tf.layers.dense({inputShape: [1], units: 1}));
  // vae.add(tf.layers.inputLayer({
  //   inputShape: [1],
  // }));
  // vae.add(tf.layers.reshape({targetShape: [1, 1, inputs.size]}));
  // vae.add(tf.layers.lstm({units: 1}));
  // vae.add(tf.layers.conv2d({filters: 5, kernelSize: 1, padding: 'same', activation: 'relu', kernelInitializer: 'randomUniform'}));
  // vae.add(tf.layers.reshape({targetShape: [1]}));
  // vae.add(tf.layers.flatten());
  vae.add(tf.layers.dense({units: 1}));
  // vae.add(tf.layers.dense({units: 1, useBias: true}));

  // const vae = tf.model({
  //   name: 'vae',
  //   inputs: encoder.inputs,
  //   outputs: decoder.apply(encoder.outputs[0])
  // });
  console.log('VAE', vae);

  // Train the model
  const tensorData = _trainModel(vae, source);

  // TODO: In LoudML upper_* and lower_* calculated as +-3 standard deviation per prediacted value
  // fill between plot: http://www.jqplot.com/deploy/dist/examples/fillBetweenLines.html

  return { vae, tensorData };
}

// ---------------------------------------------------------------------------

interface GraphPanelControllerAPI {
  series: GraphSeriesXY[];
  onSeriesAxisToggle: SeriesAxisToggleHandler;
  onSeriesColorChange: SeriesColorChangeHandler;
  onSeriesToggle: (label: string, event: React.MouseEvent<HTMLElement>) => void;
  onToggleSort: (sortBy: string) => void;
  onHorizontalRegionSelected: (from: number, to: number) => void;
}

interface GraphPanelControllerProps {
  children: (api: GraphPanelControllerAPI) => JSX.Element;
  options: Options;
  data: PanelData;
  timeZone: TimeZone;
  onOptionsChange: (options: Options) => void;
  onChangeTimeRange: (timeRange: AbsoluteTimeRange) => void;
}

interface GraphPanelControllerState {
  graphSeriesModel: GraphSeriesXY[];
}

export class GraphPanelController extends React.Component<GraphPanelControllerProps, GraphPanelControllerState> {
  constructor(props: GraphPanelControllerProps) {
    super(props);

    this.onSeriesColorChange = this.onSeriesColorChange.bind(this);
    this.onSeriesAxisToggle = this.onSeriesAxisToggle.bind(this);
    this.onToggleSort = this.onToggleSort.bind(this);
    this.onHorizontalRegionSelected = this.onHorizontalRegionSelected.bind(this);

    this.state = {
      graphSeriesModel: getGraphSeriesModel(
        props.data.series,
        props.timeZone,
        props.options.series,
        props.options.graph,
        props.options.legend,
        props.options.fieldOptions
      )
    };
  }

  static getDerivedStateFromProps(props: GraphPanelControllerProps, state: GraphPanelControllerState) {
    return {
      ...state,
      graphSeriesModel: getGraphSeriesModel(
        props.data.series,
        props.timeZone,
        props.options.series,
        props.options.graph,
        props.options.legend,
        props.options.fieldOptions
      ),
    };
  }

  onSeriesOptionsUpdate(label: string, optionsUpdate: SeriesOptions) {
    const { onOptionsChange, options } = this.props;
    const updatedSeriesOptions: { [label: string]: SeriesOptions } = { ...options.series };
    updatedSeriesOptions[label] = optionsUpdate;
    onOptionsChange({
      ...options,
      series: updatedSeriesOptions,
    });
  }

  onSeriesAxisToggle(label: string, yAxis: number) {
    const {
      options: { series },
    } = this.props;
    const seriesOptionsUpdate: SeriesOptions = series[label]
      ? {
          ...series[label],
          yAxis: {
            ...series[label].yAxis,
            index: yAxis,
          },
        }
      : {
          yAxis: {
            index: yAxis,
          },
        };
    this.onSeriesOptionsUpdate(label, seriesOptionsUpdate);
  }

  onSeriesColorChange(label: string, color: string) {
    const {
      options: { series },
    } = this.props;
    const seriesOptionsUpdate: SeriesOptions = series[label]
      ? {
          ...series[label],
          color,
        }
      : {
          color,
        };

    this.onSeriesOptionsUpdate(label, seriesOptionsUpdate);
  }

  onToggleSort(sortBy: string) {
    const { onOptionsChange, options } = this.props;
    onOptionsChange({
      ...options,
      legend: {
        ...options.legend,
        sortBy,
        sortDesc: sortBy === options.legend.sortBy ? !options.legend.sortDesc : false,
      },
    });
  }

  onHorizontalRegionSelected(from: number, to: number) {
    const { onChangeTimeRange } = this.props;
    onChangeTimeRange({ from, to });
  }

  render() {
    const { children } = this.props;
    const { graphSeriesModel } = this.state;
    const panelChrome = this._reactInternalFiber._debugOwner._debugOwner._debugOwner.stateNode;

    return (
      <GraphSeriesToggler series={graphSeriesModel}>
        {({ onSeriesToggle, toggledSeries }) => {
          return children({
            series: toggledSeries,
            onSeriesColorChange: this.onSeriesColorChange,
            onSeriesAxisToggle: this.onSeriesAxisToggle,
            onToggleSort: this.onToggleSort,
            onSeriesToggle: onSeriesToggle,
            onHorizontalRegionSelected: this.onHorizontalRegionSelected,
            panelChrome: panelChrome,
          });
        }}
      </GraphSeriesToggler>
    );
  }
}

export class LoudMLTooltip extends React.Component {
  constructor(props: any) {
    super(props);
    window.console.log('LoudMLTooltip init', props);
  }

  render () {
    const feature = (
      (
        this.props.data.request.targets
        &&this.props.data.request.targets.length>0
        &&extract_tooltip_feature(this.props.data.request.targets[0])
      )
    )|| 'Select one field'

    return (
      <div className='small'>
        <p>Use your current data selection to baseline normal metric behavior using a machine learning task.
          <br />
          This will create a new model, and run training to fit the baseline to your data.
          <br />
          You can visualise the baseline, and forecast future data using the TensorFlow tools once training is completed.
        </p>
        <p>
          <b>Feature:</b>
          <br />
          <code>{feature}</code>
        </p>
      </div>
    )
  }
}

export class CreateBaselineButton extends React.Component<> {
  constructor(props: any) {
    super(props);
    window.console.log('CreateBaselineButton init', props);
  }

  componentDidUpdate(prevProps) {
    // window.console.log('BaselineButton update', this.data);
  }

  onCreateBaselineClick() {
    if (!this.props.data.series) {
      appEvents.emit(AppEvents.alertError, ['Data Series missing. In Query settings please choose a metric']);
      return
    }

    const source = this.props.data.request.targets[0];
    const series= this.props.data.series[0];

    this.props.panelOptions.modelName = [
        extract_model_measurement(source),
        extract_model_select(source),
        extract_model_tags(source),
        extract_model_time_format(source),
    ].join('_').replace(/\./g, "_")
    this.props.panelOptions.model = _createAndTrainModel(series);
    this.props.onOptionsChange(this.props.panelOptions);
  }

  render () {
    const data = this.props.data;

    return(
      <>
      <Button size="sm" className="btn btn-inverse"
        onClick={this.onCreateBaselineClick.bind(this)}>
        <i className="fa fa-graduation-cap fa-fw"></i>
        Create TensorFlow Model
      </Button>
      <Tooltip placement="top" content={<LoudMLTooltip data={data} />}>
        <span className="gf-form-help-icon">
          <i className="fa fa-info-circle" />
        </span>
      </Tooltip>
      </>
    )
  }
}

export class MLModelController extends React.Component {
  is_trained: boolean;
  is_running: boolean;
  model: any;

  constructor(props: any) {
    super(props);
    window.console.log('MLModelController init', props);
  }

  componentDidUpdate(prevProps) {
    window.console.log('MLModelController update', this.props);
  }

  componentDidMount() {
    // this.intervalId = setInterval(this.getModel.bind(this), 5000);
  }

  componentWillUnmount() {
    // clearInterval(this.intervalId);
  }

  getModel() {
    if (!this.props.panelOptions.model) {
      return
    }

    this.model = this.props.panelOptions.model;
    window.console.log("ML getModel", this.model);

    // TODO: Update buttons based on model state
  }

  toggleModelRun() {
    // TODO
  }

  trainModel() {
    if (this.props.panelOptions.model) {
      _trainModel(this.props.panelOptions.model.vae, this.props.data.series[0]);
      this.props.onOptionsChange(this.props.panelOptions);
    }
  }

  forecastModel() {
    if (this.props.panelOptions.model) {
      const res = _forecastModel(this.props.panelOptions.model);
      console.log(res);
      this.props.onOptionsChange(this.props.panelOptions);
    }
  }

  render () {
    if (this.props.panelOptions.modelName) {
      return(
        <span className="panel-time-info">
          ML Model: {this.props.panelOptions.modelName}
          <a href="#" onClick={this.trainModel.bind(this)}> <i className="fa fa-clock-o"></i> Re-train</a>
          <a href="#" onClick={this.forecastModel.bind(this)}> <i className="fa fa-clock-o"></i> Forecast</a>

          <Tooltip placement="top" content="Current time range selection will be used to Train / Forecast">
            <span className="gf-form-help-icon">
              <i className="fa fa-info-circle" />
            </span>
          </Tooltip>
        </span>
      )
    } else {
      return null
    }
  }
}
