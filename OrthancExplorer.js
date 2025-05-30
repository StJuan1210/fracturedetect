/**
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 **/

var fractureButton = null;

$('#instance').live('pagecreate', function() {
  fractureButton = $('<a>')
      .attr('data-role', 'button')
      .attr('href', '#')
      .attr('data-icon', 'search')
      .attr('data-theme', 'e')
      .text('Deep learning for fracture');

  fractureButton.insertBefore($('#instance-delete').parent().parent());

  fractureButton.click(function() {
    if ($.mobile.pageData) {
      $.ajax({
        type: 'POST',
        url: '../fracture-apply',
        dataType: 'json',
        contentType: 'application/json',
        data: JSON.stringify({
          'instance' : $.mobile.pageData.uuid,
        }),
        cache: false,
        success: function(result) {
          console.log(result);
          window.location.assign('explorer.html#series?uuid=' + result.ParentSeries);
        },
        error: function() {
          alert('Cannot apply deep learning model');
        }
      });
    }
  });
});

$('#instance').live('pagebeforeshow', function() {
  fractureButton.hide();
});

$('#instance').live('pageshow', function() {
  $.ajax({
    url: '../instances/' + $.mobile.pageData.uuid + '/tags?simplify',
    dataType: 'json',
    cache: false,
    success: function(tags) {
      if (tags.Modality == 'XR' || tags.Modality == 'MG') {
        fractureButton.show();
      }
    }
  });
});


$('#study').live('pagebeforecreate', function() {
  var b = $('<a>')
      .attr('data-role', 'button')
      .attr('href', '#')
      .attr('data-icon', 'search')
      .attr('data-theme', 'e')
      .text('Stone Web Viewer (for fracture)');

  b.insertBefore($('#study-delete').parent().parent());
  b.click(function() {
    if ($.mobile.pageData) {
      $.ajax({
        url: '../studies/' + $.mobile.pageData.uuid,
        dataType: 'json',
        cache: false,
        success: function(study) {
          var studyInstanceUid = study.MainDicomTags.StudyInstanceUID;
          window.open('../fracture-viewer/index.html?study=' + studyInstanceUid);
        }
      });
    }
  });
});


$('#series').live('pagebeforecreate', function() {
  var b = $('<a>')
      .attr('data-role', 'button')
      .attr('href', '#')
      .attr('data-icon', 'search')
      .attr('data-theme', 'e')
      .text('Stone Web Viewer (for fracture)');

  b.insertBefore($('#series-delete').parent().parent());
  b.click(function() {
    if ($.mobile.pageData) {
      $.ajax({
        url: '../series/' + $.mobile.pageData.uuid,
        dataType: 'json',
        cache: false,
        success: function(series) {
          $.ajax({
            url: '../studies/' + series.ParentStudy,
            dataType: 'json',
            cache: false,
            success: function(study) {
              var studyInstanceUid = study.MainDicomTags.StudyInstanceUID;
              var seriesInstanceUid = series.MainDicomTags.SeriesInstanceUID;
              window.open('../fracture-viewer/index.html?study=' + studyInstanceUid +
                          '&series=' + seriesInstanceUid);
            }
          });
        }
      });
    }
  });
});

var lcancerButton = null;

$('#instance').live('pagecreate', function() {
  lcancerButton = $('<a>')
      .attr('data-role', 'button')
      .attr('href', '#')
      .attr('data-icon', 'search')
      .attr('data-theme', 'e')
      .text('Deep learning for lung cancer');

  lcancerButton.insertBefore($('#instance-delete').parent().parent());

  lcancerButton.click(function() {
    if ($.mobile.pageData) {
      $.ajax({
        type: 'POST',
        url: '../lcancer-apply',
        dataType: 'json',
        contentType: 'application/json',
        data: JSON.stringify({
          'instance' : $.mobile.pageData.uuid,
        }),
        cache: false,
        success: function(result) {
          console.log(result);
          window.location.assign('explorer.html#series?uuid=' + result.ParentSeries);
        },
        error: function() {
          alert('Cannot apply deep learning model');
        }
      });
    }
  });
});

$('#instance').live('pagebeforeshow', function() {
  lcancerButton.hide();
});

$('#instance').live('pageshow', function() {
  $.ajax({
    url: '../instances/' + $.mobile.pageData.uuid + '/tags?simplify',
    dataType: 'json',
    cache: false,
    success: function(tags) {
      if (tags.Modality == 'CT' || tags.Modality == 'MG') {
        lcancerButton.show();
      }
    }
  });
});


$('#study').live('pagebeforecreate', function() {
  var b = $('<a>')
      .attr('data-role', 'button')
      .attr('href', '#')
      .attr('data-icon', 'search')
      .attr('data-theme', 'e')
      .text('Stone Web Viewer (for lung cancer)');

  b.insertBefore($('#study-delete').parent().parent());
  b.click(function() {
    if ($.mobile.pageData) {
      $.ajax({
        url: '../studies/' + $.mobile.pageData.uuid,
        dataType: 'json',
        cache: false,
        success: function(study) {
          var studyInstanceUid = study.MainDicomTags.StudyInstanceUID;
          window.open('../lcancer-viewer/index.html?study=' + studyInstanceUid);
        }
      });
    }
  });
});


$('#series').live('pagebeforecreate', function() {
  var b = $('<a>')
      .attr('data-role', 'button')
      .attr('href', '#')
      .attr('data-icon', 'search')
      .attr('data-theme', 'e')
      .text('Stone Web Viewer (for lung cancer)');

  b.insertBefore($('#series-delete').parent().parent());
  b.click(function() {
    if ($.mobile.pageData) {
      $.ajax({
        url: '../series/' + $.mobile.pageData.uuid,
        dataType: 'json',
        cache: false,
        success: function(series) {
          $.ajax({
            url: '../studies/' + series.ParentStudy,
            dataType: 'json',
            cache: false,
            success: function(study) {
              var studyInstanceUid = study.MainDicomTags.StudyInstanceUID;
              var seriesInstanceUid = series.MainDicomTags.SeriesInstanceUID;
              window.open('../lcancer-viewer/index.html?study=' + studyInstanceUid +
                          '&series=' + seriesInstanceUid);
            }
          });
        }
      });
    }
  });
});
