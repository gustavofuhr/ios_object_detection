//
//  ViewController.swift
//  RTMDetObjectDetection
//
//  Created by Gustavo Führ on 19/07/24.
//

import UIKit

var mlModel = try! rtmdet_tiny(configuration: .init()).model


class ViewController: UIViewController {

    private var videoCapture: VideoCapture!
    
    private let videoSize = CGSize(width: 1280, height: 720)
    private let preferredFps: Int32 = 2

    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var bbView: BoundingBoxView!

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        let spec = VideoSpec(fps: preferredFps, size: videoSize)
        let frameInterval = 1.0 / Double(preferredFps)
        
        videoCapture = VideoCapture(cameraType: .back,
                                    preferredSpec: spec,
                                    previewContainer: cameraView.layer)
        videoCapture.imageBufferHandler = {[unowned self] (imageBuffer, timestamp, outputBuffer) in
            let delay = CACurrentMediaTime() - timestamp.seconds
            if delay > frameInterval {
                return
            }

            self.serialQueue.async {
                self.runModel(imageBuffer: imageBuffer)
            }
        }
        
        let modelPaths = Bundle.main.paths(forResourcesOfType: "mlmodel", inDirectory: "models")
        
        modelUrls = []
        for modelPath in modelPaths {
            let url = URL(fileURLWithPath: modelPath)
            let compiledUrl = try! MLModel.compileModel(at: url)
            modelUrls.append(compiledUrl)
        }
        
        selectModel(url: modelUrls.first!)
        
        // scaleFill
        cropAndScaleOptionSelector.selectedSegmentIndex = 2
        updateCropAndScaleOption()
        

    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        guard let videoCapture = videoCapture else {return}
        videoCapture.startCapture()
    }
    
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        guard let videoCapture = videoCapture else {return}
        videoCapture.resizePreview()
        // TODO: Should be dynamically determined
        self.bbView.updateSize(for: CGSize(width: videoSize.height, height: videoSize.width))
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        guard let videoCapture = videoCapture else {return}
        videoCapture.stopCapture()
        super.viewWillDisappear(animated)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    


}

